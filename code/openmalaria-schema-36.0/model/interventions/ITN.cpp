/* This file is part of OpenMalaria.
 * 
 * Copyright (C) 2005-2015 Swiss Tropical and Public Health Institute
 * Copyright (C) 2005-2015 Liverpool School Of Tropical Medicine
 * 
 * OpenMalaria is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

#include "interventions/ITN.h"
#include "util/random.h"
#include "util/errors.h"
#include "util/SpeciesIndexChecker.h"
#include "Host/Human.h"
#include "R_nmath/qnorm.h"
#include <cmath>

namespace OM { namespace interventions {

vector<ITNComponent*> ITNComponent::componentsByIndex;

ITNComponent::ITNComponent( ComponentId id, const scnXml::ITNDescription& elt,
        const map<string, size_t>& species_name_map ) :
        Transmission::HumanVectorInterventionComponent(id),
        ripFactor( numeric_limits<double>::signaling_NaN() )
{
    initialInsecticide.setParams( elt.getInitialInsecticide() );
    const double maxProp = 0.999;       //NOTE: this could be exposed in XML, but probably doesn't need to be
    maxInsecticide = R::qnorm5(maxProp, initialInsecticide.getMu(), initialInsecticide.getSigma(), true, false);
    holeRate.setParams( elt.getHoleRate() );    // per year
    holeRate.scaleMean( sim::yearsPerStep() );  // convert to per step
    ripRate.setParams( elt.getRipRate() );
    ripRate.scaleMean( sim::yearsPerStep() );
    ripFactor = elt.getRipFactor().getValue();
    insecticideDecay = DecayFunction::makeObject( elt.getInsecticideDecay(), "ITNDescription.insecticideDecay" );
    attritionOfNets = DecayFunction::makeObject( elt.getAttritionOfNets(), "ITNDescription.attritionOfNets" );
    // assume usage modifier is 100% if none is specified
    double propUse;
    if (elt.getUsage().present()) {
        propUse = elt.getUsage().get().getValue();
    }
    else {
        propUse = 1.0;
    }
    if( !( propUse >= 0.0 && propUse <= 1.0 ) ){
        throw util::xml_scenario_error("ITN.description.proportionUse: must be within range [0,1]");
    }
    
    typedef scnXml::ITNDescription::AnophelesParamsSequence AP;
    const AP& ap = elt.getAnophelesParams();
    species.resize(species_name_map.size());
    util::SpeciesIndexChecker checker( "ITN", species_name_map );
    for( AP::const_iterator it = ap.begin(); it != ap.end(); ++it ){
        species[checker.getIndex(it->getMosquito())].init (*it, propUse, maxInsecticide);
    }
    checker.checkNoneMissed();
    
    if( componentsByIndex.size() <= id.id ) componentsByIndex.resize( id.id+1, 0 );
    componentsByIndex[id.id] = this;
}

void ITNComponent::deploy( Host::Human& human, mon::Deploy::Method method, VaccineLimits )const{
    human.perHostTransmission.deployComponent( *this );
    mon::reportEventMHD( mon::MHD_ITN, human, method );
}

Component::Type ITNComponent::componentType() const{
    return Component::ITN;
}
    
#ifdef WITHOUT_BOINC
void ITNComponent::print_details( std::ostream& out )const{
    out << id().id << "\tITN";
}
#endif

PerHostInterventionData* ITNComponent::makeHumanPart() const{
    return new HumanITN( *this );
}
PerHostInterventionData* ITNComponent::makeHumanPart( istream& stream, ComponentId id ) const{
    return new HumanITN( stream, id );
}

void ITNComponent::ITNAnopheles::init(
    const scnXml::ITNDescription::AnophelesParamsType& elt,
    double proportionUse,
    double maxInsecticide)
{
    assert( _relativeAttractiveness.get() == 0 );       // double init
    if (elt.getDeterrency().present())
        _relativeAttractiveness = boost::shared_ptr<RelativeAttractiveness>(
            new RADeterrency( elt.getDeterrency().get(), maxInsecticide ));
    else{
        assert (elt.getTwoStageDeterrency().present());
        _relativeAttractiveness = boost::shared_ptr<RelativeAttractiveness>(
            new RATwoStageDeterrency( elt.getTwoStageDeterrency().get(), maxInsecticide ));
    }
    _preprandialKillingEffect.init( elt.getPreprandialKillingEffect(),
                                    maxInsecticide,
                                    "ITN.description.anophelesParams.preprandialKillingFactor", false );
    _postprandialKillingEffect.init( elt.getPostprandialKillingEffect(),
                                    maxInsecticide,
                                    "ITN.description.anophelesParams.postprandialKillingFactor", false );
    // Nets only affect people while they're using the net. NOTE: we may want
    // to revise this at some point (heterogeneity, seasonal usage patterns).
    double propActive = elt.getPropActive();
    assert( proportionUse >= 0.0 && proportionUse <= 1.0 );
    assert( propActive >= 0.0 && propActive <= 1.0 );
    proportionProtected = proportionUse * propActive;
    proportionUnprotected = 1.0 - proportionProtected;
}

ITNComponent::ITNAnopheles::RADeterrency::RADeterrency(const scnXml::ITNDeterrency& elt,
                                                   double maxInsecticide) :
    lHF( numeric_limits< double >::signaling_NaN() ),
    lPF( numeric_limits< double >::signaling_NaN() ),
    lIF( numeric_limits< double >::signaling_NaN() ),
    holeScaling( numeric_limits< double >::signaling_NaN() ),
    insecticideScaling( numeric_limits< double >::signaling_NaN() )
{
    double HF = elt.getHoleFactor();
    double PF = elt.getInsecticideFactor();
    double IF = elt.getInteractionFactor();
    holeScaling = elt.getHoleScalingFactor();
    insecticideScaling = elt.getInsecticideScalingFactor();
    if( !(holeScaling>=0.0 && insecticideScaling>=0.0) ){
        throw util::xml_scenario_error("ITN.description.anophelesParams.deterrency: expected scaling factors to be non-negative");
    }
    
    /* We need to ensure the relative availability is non-negative. However,
     * since it's an exponentiated value, it always will be.
     * 
     * If don't want nets to be able to increase transmission, the following
     * limits could also be applied. In general, however, there is no reason
     * nets couldn't make individuals more attractive to mosquitoes.
     * 
     * To ensure relative availability is at most one: relative availability is
     *  exp( log(HF)*h + log(PF)*p + log(IF)*h*p )
     * where HF, PF and IF are the hole, insecticide and interaction factors
     * respectively, with h and p defined as:
     *  h=exp(-holeIndex*holeScalingFactor),
     *  p=1−exp(-insecticideContent*insecticideScalingFactor).
     * We therefore need to ensure that:
     *  log(HF)*h + log(PF)*p + log(IF)*h*p ≤ 0
     * 
     * As with the argument below concerning limits of the killing effect
     * parameters, h and p will always be in the range [0,1] and p ≤ pmax.
     * We can then derive some bounds for HF and PF:
     *  log(HF) ≤ 0
     *  log(PF)×pmax ≤ 0
     *  log(HF) + (log(PF)+log(IF))×pmax = log(HF×(PF×IF)^pmax) ≤ 0
     * or equivalently (assuming pmax>0):
     *  HF ∈ (0,1]
     *  PF ∈ (0,1]
     *  HF×(PF×IF)^pmax ∈ (0,1]
     *
     * Weaker limits would not be sufficient, as with the argument for the
     * limits of killing effect arguments below. */
#ifdef WITHOUT_BOINC
    // Print out a warning if nets may increase transmission, but only in
    // non-BOINC mode, since it is not unreasonable and volunteers often
    // mistake this kind of warning as indicating a problem.
    double pmax = 1.0-exp(-maxInsecticide*insecticideScaling);
    if( !( HF > 0.0 && PF > 0.0 && IF > 0.0 &&
            HF <= 1.0 && PF <= 1.0 && HF*pow(PF*IF,pmax) <= 1.0 ) )
    {
        cerr << "Note: since the following bounds are not met, the ITN could make humans more\n";
        cerr << "attractive to mosquitoes than they would be without a net.\n";
        cerr << "This note is only shown by non-BOINC executables.\n";
        cerr << "ITN.description.anophelesParams.deterrency: bounds not met:\n";
        if( !(HF>0.0) )
            cerr << "  holeFactor>0\n";
        if( !(PF>0.0) )
            cerr << "  insecticideFactor>0\n";
        if( !(IF>0.0) )
            cerr << "  interactionFactor>0\n";
        if( !(HF<=1.0) )
            cerr << "  holeFactor≤1\n";
        if( !(PF<=1.0) )
            cerr << "  insecticideFactor≤1\n";
        if( !(HF*pow(PF*IF,pmax)<=1.0) )
            cerr << "  holeFactor×(insecticideFactor×interactionFactor)^"<<pmax<<"≤1\n";
        cerr.flush();
    }
#endif
    lHF = log( HF );
    lPF = log( PF );
    lIF = log( IF );
}
ITNComponent::ITNAnopheles::RATwoStageDeterrency::RATwoStageDeterrency(
        const scnXml::TwoStageDeterrency& elt, double maxInsecticide) :
    lPFEntering( numeric_limits< double >::signaling_NaN() ),
    insecticideScalingEntering( numeric_limits< double >::signaling_NaN() )
{
    double PF = elt.getEntering().getInsecticideFactor();
    insecticideScalingEntering = elt.getEntering().getInsecticideScalingFactor();
    if( !( PF > 0.0) ){
        // we take the log of PF, so it must be positive
        ostringstream msg;
        msg << "ITN.description.anophelesParams.twoStageDeterrency.entering: insecticideFactor must be positive since we take its logarithm.";
        throw util::xml_scenario_error( msg.str() );
    }
    
    /* We need to ensure the relative availability is non-negative. However,
     * since it's an exponentiated value, it always will be.
     * 
     * If we don't want ITNs to be able to increase transmission, the following
     * limits could also be applied. In general, however, there is no reason
     * ITNs couldn't make individuals more attractive to mosquitoes.
     * 
     * To ensure relative availability is at most one: relative availability is
     *  exp( log(PF)*p ) = PF^p
     * where PF is the insecticide factor, with p∈[0,1] defined as:
     *  p=1−exp(-insecticideContent*insecticideScalingFactor).
     * We therefore just need PF ≤ 1. */
#ifdef WITHOUT_BOINC
    // Print out a warning if ITNs may increase transmission, but only in
    // non-BOINC mode, since it is not unreasonable and volunteers often
    // mistake this kind of warning as indicating a problem.
    if( !( PF <= 1.0 ) ) {
        cerr << "Note: since the following bounds are not met, the IRS could make humans more\n";
        cerr << "attractive to mosquitoes than they would be without IRS.\n";
        cerr << "This note is only shown by non-BOINC executables.\n";
        cerr << "IRS.description.anophelesParams.deterrency: bounds not met:\n";
        cerr << "  0<insecticideFactor≤1\n";
        cerr.flush();
    }
#endif
    lPFEntering = log( PF );
    
    pAttacking.init( elt.getAttacking(), maxInsecticide, "ITN.description.anophelesParams.twoStageDeterrency.attacking", true );
}
ITNComponent::ITNAnopheles::SurvivalFactor::SurvivalFactor() :
    BF( numeric_limits< double >::signaling_NaN() ),
    HF( numeric_limits< double >::signaling_NaN() ),
    PF( numeric_limits< double >::signaling_NaN() ),
    IF( numeric_limits< double >::signaling_NaN() ),
    holeScaling( numeric_limits< double >::signaling_NaN() ),
    insecticideScaling( numeric_limits< double >::signaling_NaN() ),
    invBaseSurvival( numeric_limits< double >::signaling_NaN() )
{}
void ITNComponent::ITNAnopheles::SurvivalFactor::init(const scnXml::ITNKillingEffect& elt,
                                                   double maxInsecticide, const char* eltName,
                                                   bool raTwoStageConstraints){
    BF = elt.getBaseFactor();
    HF = elt.getHoleFactor();
    PF = elt.getInsecticideFactor();
    IF = elt.getInteractionFactor();
    holeScaling = elt.getHoleScalingFactor();
    insecticideScaling = elt.getInsecticideScalingFactor();
    invBaseSurvival = 1.0 / (1.0 - BF);
    if( !( BF >= 0.0 && BF < 1.0) ){
        ostringstream msg;
        msg << eltName << ": expected baseFactor to be in range [0,1]";
        throw util::xml_scenario_error( msg.str() );
    }
    if( !(holeScaling>=0.0 && insecticideScaling>=0.0) ){
        ostringstream msg;
        msg << eltName << ": expected scaling factors to be non-negative";
        throw util::xml_scenario_error( msg.str() );
    }
    // see below
    double pmax = 1.0-exp(-maxInsecticide*insecticideScaling);
    
    if( raTwoStageConstraints ){
        // Note: the following argument is a modification of the one below
        // (when !raTwoStageConstraints). The original may make more sense.
    /* We want K ≥ 0 where K is the killing factor:
    K=BF+HF×h+PF×p+IF×h×p, with h and p defined as:
    h=exp(-holeIndex×holeScalingFactor),
    p=1−exp(-insecticideContent×insecticideScalingFactor). 
    
    By their nature, holeIndex ≥ 0 and insecticideContent ≥ 0. We restrict:
        holeScalingFactor ≥ 0
        insecticideScalingFactor ≥ 0
    Which implies both h and p lie in the range [0,1]. We also know 0 ≤ BF ≤ 1.
    
    We need K ≥ 0 or:
        BF + HF×h + PF×p + IF×h×p ≥ 0   (1)
    
    Lets derive some limits on HF, PF and IF such that the above inequality (1)
    is satisfied.
    
    A net can theoretically be unholed (holeIndex=0 ⇒ h=1) and have no
    insecticide (thus have p=0). Substituting these values in (1) yields:
        BF + HF ≥ 0     (2)
    
    The maximum value for p depends on the maximum insecticide content; denote
    pmax = max(p). Note that holeIndex has no finite maximum; thus, although
    for any finite value of holeIndex, h > 0, there is no h₀ > 0 s.t. for all
    values of holeIndex h ≥ h₀. For the limiting case of a tattered but
    insecticide-saturated net our parameters are therefore p=pmax, h=0:
        BF + PF×pmax ≥ 0        (3)
    
    Consider a net saturated with insecticide (p=pmax) and without holes (h=1):
        BF + HF + (PF+IF)×pmax ≥ 0      (4)
    
    The opposite extreme (the limiting case of a decayed net with no remaining
    insecticide and a large number of holes) yields only BF ≥ 0.
    
    Some of the above examples of nets may be unlikely, but there is only one
    restriction in our model making any of these cases impossible: some
    insecticide must have been lost by the time any holes occur. We ignore this
    since its effect is likely small, and thus all of the above are required to
    keep the factor in the range [0,1]. Further, the inequalities (2) - (3) are
    sufficient to keep the factor within [0,1] since h and p are non-negative
    and act linearly in (1).
    
    From the definition of p, we always have pmax ≤ 1, so substituting pmax=1
    in (5) - (8) gives us bounds which imply our requirement, however if pmax
    is finite they are stricter than necessary. Since insecticideScalingFactor
    is constant, max(p) coincides with max(insecticideContent) which, since
    insecticide content only decays over time, coincides with the maximum
    initial insecticide content, Pmax. Since the initial insecticide content is
    sampled from a normal distribution in our model it should have no finite
    maximum, thus implying we cannot achieve more relaxed bounds than (5) - (8)
    when pmax=1 (unless the standard deviation of our normal distribution is 1).
    We would however like to impose less strict bounds than these, thus we
    impose a maximum value on the initial insecticide content, Pmax, such that
    the probability of sampling a value from our parameterise normal
    distribution greater than Pmax is 0.001. */
        if( !( BF+HF >= 0.0
            && BF+PF*pmax >= 0.0
            && BF+HF+(PF+IF)*pmax >= 0.0 ) )
        {
            ostringstream msg;
            msg << eltName << ": bounds not met:";
            if( !(BF+HF >= 0.0) )
                msg << " baseFactor+holeFactor≥0";
            if( !(BF+PF*pmax >= 0.0) )
                msg << " baseFactor+"<<pmax<<"×insecticideFactor≥0";
            if( !(PF+HF+(PF+IF)*pmax >= 0.0) )
                msg << " baseFactor+holeFactor+"<<pmax<<"×(insecticideFactor+interactionFactor)≥0";
            throw util::xml_scenario_error( msg.str() );
        }
    }else{
    /* We want the calculated survival factor (1−K)/(1−BF) to be in the range
    [0,1] where K is the killing factor: K=BF+HF×h+PF×p+IF×h×p, with h and p
    defined as: h=exp(-holeIndex×holeScalingFactor),
    p=1−exp(-insecticideContent×insecticideScalingFactor). 
    
    By their nature, holeIndex ≥ 0 and insecticideContent ≥ 0. We restrict:
        holeScalingFactor ≥ 0
        insecticideScalingFactor ≥ 0
    Which implies both h and p lie in the range [0,1]. We also know the base
    survival factor, 1−BF, is in the range [0,1].
    
    To make sure the survival factor is not negative we need (1−K)/(1−BF) ≥ 0.
    Since 1−BF > 0 we need 1−K ≥ 0, which, substituting K, gives us
        BF + HF×h + PF×p + IF×h×p ≤ 1	(1)
    We also want to make sure the survival factor is not greater than one (since
    nets shouldn't increase mosquito survival), (1−K)/(1−BF) ≤ 1 or equivalently
    1-K ≤ 1-BF or K ≥ BF, which, substituting K, yields
        HF×h + PF×p + IF×h×p ≥ 0		(2)
    
    Lets derive some limits on HF, PF and IF such that the above inequalities
    (1) and (2) are satisfied.
    
    A net can theoretically be unholed (holeIndex=0 ⇒ h=1) and have no
    insecticide (thus have p=0). Substituting these values in (1) and (2) yields:
        BF + HF ≤ 1	(3)
        HF ≥ 0		(4)
    
    The maximum value for p depends on the maximum insecticide content; denote
    pmax = max(p). Note that holeIndex has no finite maximum; thus, although
    for any finite value of holeIndex, h > 0, there is no h₀ > 0 s.t. for all
    values of holeIndex h ≥ h₀. For the limiting case of a tattered but
    insecticide-saturated net our parameters are therefore p=pmax, h=0:
        BF + PF×pmax ≤ 1	(5)
        PF×pmax ≥ 0			(6)
    (Assuming pmax > 0, (6) is equivalent to PF ≥ 0.)
    
    Consider a net saturated with insecticide (p=pmax) and without holes (h=1):
        BF + HF + (PF+IF)×pmax ≤ 1	(7)
        HF + (PF+IF)×pmax ≥ 0		(8)
    
    The opposite extreme (the limiting case of a decayed net with no remaining
    insecticide and a large number of holes) yields only BF ≤ 1 which we already
    know.
    
    Some of the above examples of nets may be unlikely, but there is only one
    restriction in our model making any of these cases impossible: some
    insecticide must have been lost by the time any holes occur. We ignore this
    since its effect is likely small, and thus all of the above are required to
    keep the survival factor in the range [0,1]. Further, these six inequalities
    (3) - (8) are sufficient to keep the survival factor within [0,1] since h
    and p are non-negative and act linearly in (1) and (2).
    
    From the definition of p, we always have pmax ≤ 1, so substituting pmax=1
    in (5) - (8) gives us bounds which imply our requirement, however if pmax
    is finite they are stricter than necessary. Since insecticideScalingFactor
    is constant, max(p) coincides with max(insecticideContent) which, since
    insecticide content only decays over time, coincides with the maximum
    initial insecticide content, Pmax. Since the initial insecticide content is
    sampled from a normal distribution in our model it should have no finite
    maximum, thus implying we cannot achieve more relaxed bounds than (5) - (8)
    when pmax=1 (unless the standard deviation of our normal distribution is 1).
    We would however like to impose less strict bounds than these, thus we
    impose a maximum value on the initial insecticide content, Pmax, such that
    the probability of sampling a value from our parameterise normal
    distribution greater than Pmax is 0.001. */
        if( !( BF+HF <= 1.0 && HF >= 0.0
            && BF+PF*pmax <= 1.0 && PF >= 0.0
            && BF+HF+(PF+IF)*pmax <= 1.0 && HF+(PF+IF)*pmax >= 0.0 ) )
        {
            ostringstream msg;
            msg << eltName << ": bounds not met:";
            if( !(BF+HF<=1.0) )
                msg << " baseFactor+holeFactor≤1";
            if( !(HF>=0.0) )
                msg << " holeFactor≥0";
            if( !(BF+PF*pmax<=1.0) )
                msg << " baseFactor+"<<pmax<<"×insecticideFactor≤1";
            if( !(PF>=0.0) )
                msg << " insecticideFactor≥0";      // if this fails, we know pmax>0 (since it is in any case non-negative) — well, or an NaN
            if( !(PF+HF+(PF+IF)*pmax<=1.0) )
                msg << " baseFactor+holeFactor+"<<pmax<<"×(insecticideFactor+interactionFactor)≤1";
            if( !(HF+(PF+IF)*pmax>=0.0) )
                msg << " holeFactor+"<<pmax<<"×(insecticideFactor+interactionFactor)≥0";
            throw util::xml_scenario_error( msg.str() );
        }
    }
}
double ITNComponent::ITNAnopheles::RADeterrency::relativeAttractiveness( double holeIndex, double insecticideContent )const {
    double holeComponent = exp(-holeIndex*holeScaling);
    double insecticideComponent = 1.0 - exp(-insecticideContent*insecticideScaling);
    double relAvail = exp( lHF*holeComponent + lPF*insecticideComponent + lIF*holeComponent*insecticideComponent );
    assert( relAvail>=0.0 );
    return relAvail;
}
double ITNComponent::ITNAnopheles::RATwoStageDeterrency::relativeAttractiveness(
        double holeIndex, double insecticideContent )const
{
    // This is essentially a combination of the relative attractiveness as used
    // by IRS and a killing factor.
    
    // Note that an alternative, simpler, model could have been used, but was
    // not for consistency with other models. Alternative (here we don't take
    // the logarithm of PF):
    // pEnt = 1 - PFEntering × insecticideComponent
    
    double insecticideComponent = 1.0 - exp(-insecticideContent*insecticideScalingEntering);
    double pEnt = exp( lPFEntering*insecticideComponent );
    assert( pEnt >= 0.0 );
    
    double rel_pAtt = pAttacking.rel_pAtt( holeIndex, insecticideContent );
    double relAttr = pEnt * rel_pAtt;
    assert( relAttr >= 0.0 );
    return relAttr;
}
double ITNComponent::ITNAnopheles::SurvivalFactor::rel_pAtt( double holeIndex, double insecticideContent )const {
    double holeComponent = exp(-holeIndex*holeScaling);
    double insecticideComponent = 1.0 - exp(-insecticideContent*insecticideScaling);
    double pAtt = BF + HF*holeComponent + PF*insecticideComponent + IF*holeComponent*insecticideComponent;
    assert( pAtt >= 0.0 );
    return pAtt / BF;
}
double ITNComponent::ITNAnopheles::SurvivalFactor::survivalFactor( double holeIndex, double insecticideContent )const {
    double holeComponent = exp(-holeIndex*holeScaling);
    double insecticideComponent = 1.0 - exp(-insecticideContent*insecticideScaling);
    double killingEffect = BF + HF*holeComponent + PF*insecticideComponent + IF*holeComponent*insecticideComponent;
    double survivalFactor = (1.0 - killingEffect) * invBaseSurvival;
    assert( killingEffect <= 1.0 );
    // survivalFactor might be out of bounds due to precision error, see #49
    if (survivalFactor < 0.0)
        return 0.0;
    else if (survivalFactor > 1.0)
        return 1.0;
    return survivalFactor;
}

HumanITN::HumanITN( const ITNComponent& params ) :
        PerHostInterventionData( params.id() ),
        nHoles( 0 ),
        holeIndex( 0.0 )
{
    // Net rips and insecticide loss are assumed to co-vary dependent on
    // handling of net. They are sampled once per human: human handling is
    // presumed to be the largest cause of variance.
    util::NormalSample x = util::NormalSample::generate();
    holeRate = params.holeRate.sample(x);
    ripRate = params.ripRate.sample(x);
    insecticideDecayHet = params.insecticideDecay->hetSample(x);

    // Sample per-deployment variables as in redeploy:
    disposalTime = sim::now() + params.attritionOfNets->sampleAgeOfDecay();
    // this is sampled independently: initial insecticide content doesn't depend on handling
    initialInsecticide = params.initialInsecticide.sample();
    if( initialInsecticide < 0.0 )
        initialInsecticide = 0.0;       // avoid negative samples
    if( initialInsecticide > params.maxInsecticide )
        initialInsecticide = params.maxInsecticide;
}

void HumanITN::redeploy(const OM::Transmission::HumanVectorInterventionComponent& params0) {
    const ITNComponent& params = *dynamic_cast<const ITNComponent*>(&params0);
    
    deployTime = sim::nowOrTs1();
    disposalTime = sim::nowOrTs1() + params.attritionOfNets->sampleAgeOfDecay();
    nHoles = 0;
    holeIndex = 0.0;
    // this is sampled independently: initial insecticide content doesn't depend on handling
    initialInsecticide = params.initialInsecticide.sample();
    if( initialInsecticide < 0.0 )
        initialInsecticide = 0.0;	// avoid negative samples
    if( initialInsecticide > params.maxInsecticide )
        initialInsecticide = params.maxInsecticide;
}

void HumanITN::update(Host::Human& human){
    const ITNComponent& params = *ITNComponent::componentsByIndex[m_id.id];
    if( deployTime != sim::never() ){
        // First use is at age 0 relative to ts0()
        if( sim::ts0() >= disposalTime ){
            deployTime = sim::never();
            human.removeFromSubPop(id());
            return;
        }
        
        int newHoles = util::random::poisson( holeRate );
        nHoles += newHoles;
        holeIndex += newHoles + params.ripFactor * util::random::poisson( nHoles * ripRate );
    }
}

double HumanITN::relativeAttractiveness(size_t speciesIndex) const{
    if( deployTime == sim::never() ) return 1.0;
    const ITNComponent& params = *ITNComponent::componentsByIndex[m_id.id];
    const ITNComponent::ITNAnopheles& anoph = params.species[speciesIndex];
    return anoph.relativeAttractiveness( holeIndex, getInsecticideContent(params) );
}

double HumanITN::preprandialSurvivalFactor(size_t speciesIndex) const{
    if( deployTime == sim::never() ) return 1.0;
    const ITNComponent& params = *ITNComponent::componentsByIndex[m_id.id];
    const ITNComponent::ITNAnopheles& anoph = params.species[speciesIndex];
    return anoph.preprandialSurvivalFactor( holeIndex, getInsecticideContent(params) );
}

double HumanITN::postprandialSurvivalFactor(size_t speciesIndex) const{
    if( deployTime == sim::never() ) return 1.0;
    const ITNComponent& params = *ITNComponent::componentsByIndex[m_id.id];
    const ITNComponent::ITNAnopheles& anoph = params.species[speciesIndex];
    return anoph.postprandialSurvivalFactor( holeIndex, getInsecticideContent(params) );
}

void HumanITN::checkpoint( ostream& stream ){
    deployTime & stream;
    disposalTime & stream;
    nHoles & stream;
    holeIndex & stream;
    initialInsecticide & stream;
    holeRate & stream;
    ripRate & stream;
    insecticideDecayHet & stream;
}
HumanITN::HumanITN( istream& stream, ComponentId id ) : PerHostInterventionData( id )
{
    deployTime & stream;
    disposalTime & stream;
    nHoles & stream;
    holeIndex & stream;
    initialInsecticide & stream;
    holeRate & stream;
    ripRate & stream;
    insecticideDecayHet & stream;
}

} }
