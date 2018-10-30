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

#ifndef Hmod_AnophelesModel
#define Hmod_AnophelesModel

#include "Global.h"
#include "Transmission/Anopheles/PerHost.h"
#include "Transmission/PerHost.h"
#include "Transmission/Anopheles/MosqTransmission.h"
#include "Transmission/Anopheles/FixedEmergence.h"
#include "util/SimpleDecayingValue.h"
#include "util/vectors.h"

#include <vector>
#include <limits>
#include <boost/shared_ptr.hpp>

namespace OM {
    class Population;
namespace Transmission {
namespace Anopheles {
    using std::numeric_limits;
    using util::vector2D;

/** Per-species part for vector transmission model.
 *
 * Data in this class is specific to a species of anopheles mosquito, where
 * species is used in a relaxed way to mean any variation of anopheles
 * mosquito, not just those types formally recognised as distinct species.
 *
 * A list of this class type is used by the VectorModel class to hold
 * (potentially) species-specific per-population data.
 *
 * Instead of storing static variables in this class, store them in the
 * VectorModel class.
 *
 * Variable names largely come from Nakul Chitnis's paper:
 * "A mathematical model for the dynamics of malaria in mosquitoes feeding on
 * a heterogeneous host population" (3rd Oct. 2007). */
class AnophelesModel
{
public:
    ///@brief Initialisation and destruction
    //@{
    AnophelesModel () :
            mosqSeekingDuration(numeric_limits<double>::signaling_NaN()),
            mosqSeekingDeathRate(numeric_limits<double>::signaling_NaN()),
            probMosqSurvivalOvipositing(numeric_limits<double>::signaling_NaN())
    {}
    
    /** Called to initialise variables instead of a constructor. At this point,
     * the size of the human population is known but that population has not
     * yet been constructed. Called whether data is loaded from a check-point
     * or not.
     *
     * @param anoph Data structure from XML to use
     * @param initialisationEIR In/out parameter: TransmissionModel::initialisationEIR
     * @param nonHumanHostPopulations Size of each non-human population
     * @param populationSize Size of human population (assumed constant)
     */
    string initialise (const scnXml::AnophelesParams& anoph,
                       vector<double>& initialisationEIR,
                       map<string, double>& nonHumanHostPopulations,
                       int populationSize);
    
    /** Scale the internal EIR representation by factor; used as part of
     * initialisation. */
    inline void scaleEIR( double factor ){
        transmission.emergence->scaleEIR( factor );
    }
    
    /** Initialisation which must wait until a human population is available.
     * This is only called when a checkpoint is not loaded.
     *
     * @param sIndex Index in VectorModel::species of this class.
     * @param population The human population
     * @param meanPopAvail The mean availability of age-based relative
     * availability of humans to mosquitoes across populations.
     *
     * Can only usefully run its calculations when not checkpointing, due to
     * population not being the same when loaded from a checkpoint. */
    void init2 (size_t sIndex,
                   const OM::Population& population,
                   double meanPopAvail);
    
    /** Set up the non-host-specific interventions. */
    void initVectorInterv( const scnXml::VectorSpeciesIntervention& elt, size_t instance );
    
    /** Set up trap parameters. */
    void initVectorTrap( const scnXml::Description1& desc, size_t instance );
    
    /** Return base-line human parameters for the mosquito. */
    inline const Anopheles::PerHostBase& getHumanBaseParams () {
        return humanBase;
    }
    
    /** Work out whether another interation is needed for initialisation and if
     * so, make necessary changes.
     *
     * @returns true if another iteration is needed. */
    inline bool initIterate (){
        return transmission.emergence->initIterate(transmission);
    }
    //@}

    
    ///@brief Functions called as part of usual per-time-step operations
    //@{
    /** Called per time-step. Does most of calculation of EIR.
     *
     * @param population The human population; so we can sum up availability and
     *  infectiousness.
     * @param popProbTransmission A two-dimensional vector of the probability
     *  of transmission to mosquito for each human host (first index, in same
     *  order as population) and for each parasite genotype (second index).
     *  This is simply a cache, since calling probTransmissionToMosquito() is
     *  expensive.
     * @param sIndex Index of the type of mosquito in per-type/species lists.
     * @param isDynamic True to use full model; false to drive model from current contents of S_v.
     */
    void advancePeriod( const OM::Population& population,
                        vector2D<double>& popProbTransmission,
                        size_t sIndex, bool isDynamic );

    /** Returns the EIR calculated by advancePeriod().
     *
     * Could be extended to allow input EIR driven initialisation on a per-species
     * level instead of the whole simulation, but that doesn't appear worth doing.
     *
     * @param sIndex Index of this in VectorModel::species
     * @param host PerHost of the human requesting this EIR.
     * @param EIR Vector of EIR per genotype; function adds to this to
     *  eventually yield the EIR per genotype summed across species.
     */
    void calculateEIR (size_t sIndex, Transmission::PerHost& host,
            vector<double>& EIR );
    //@}


    ///@brief Functions called to deploy interventions
    //@{
    void deployVectorPopInterv (size_t instance);
    /// Deploy some traps
    /// 
    /// @param instance Index of this type of trap
    /// @param number The number of traps to deploy
    /// @param lifespan Time until these traps are removed/replaced/useless
    void deployVectorTrap(size_t instance, double number, SimTime lifespan);

    inline void uninfectVectors() {
        transmission.uninfectVectors();
    }
    //@}

    
    ///@brief Functions used in reporting
    //@{
    /// Get mean emergence during last time-step
    inline double getLastN_v0 () const{
        return transmission.getLastN_v0();
    }
    /// Get mean P_A/P_df/P_dif/N_v/O_v/S_v during last time-step
    /// @param vs PA, PDF, PDIF, NV, OV or SV
    inline double getLastVecStat( VecStat vs )const{
        return transmission.getLastVecStat( vs );
    }
    
    inline double getResAvailability() const{
        return transmission.getResAvailability();
    }
    inline double getResRequirements() const{
        return transmission.getResRequirements();
    }

    /// Write some per-species summary information.
    inline void summarize( size_t species )const{
        transmission.summarize( species );
    }
    //@}
    

    /// Checkpointing
    //Note: below comments about what does and doesn't need checkpointing are ignored here.
    template<class S>
    void operator& (S& stream) {
        mosqSeekingDeathRate & stream;
        mosqSeekingDuration & stream;
        probMosqSurvivalOvipositing & stream;
        transmission & stream;
        seekingDeathRateIntervs & stream;
        probDeathOvipositingIntervs & stream;
        baitedTraps & stream;
        partialEIR & stream;
    }


private:
    ///@brief Initialisation helper functions
    //@{
    /** Calculate availability rate of hosts (α_i) and death rate while seeking
     * (µ_vA)
     *
     * Documentation: "Parameter Values for Transmission model"
     * (Chitnis, Smith and Schapira, 4.3.2010)
     * 
     * @param anoph Data from XML
     * @param nonHumanHostPopulations Size of each non-human population
     * @param populationSize Size of the human population (assumed constant)
     */
    void initAvailability(
        const scnXml::AnophelesParams& anoph,
        map<string, double>& nonHumanHostPopulations,
        int populationSize);

    /** Calculates the human ento availability
     * 
     * Reference: Parameter Values for Transmission Model, Chitnis et al,
     * September 2010 eqn (26).
     * 
     * @param N_i Human/non-human population size
     * @param P_A Probability of mosquito not dying or finding a host while
     *  seeking on a given night
     * @param P_Ai Probability of mosquito finding a human/non-human host of
     *  type i while seeking on a given night
     * @return α_i, the rate at which mosquitoes encounter hosts of type i
     *  while seeking
     */
    double calcEntoAvailability(double N_i, double P_A, double P_Ai);
    //@}
    
    
    // -----  parameters (constant after initialisation)  -----
    
    /** Baseline parameters which may be varied per human host. The primary
     * reason for wrapping these parameters in a struct is that these are the
     * parameters which need to be passed to the PerHost code
     * during calculations.
     *
     * Includes model parameters which may be varied per-individual to account
     * for interventions and innate resistances, and intervention effect
     * descriptions.
     *
     * Read from XML by initialise; no need to checkpoint. */
    Anopheles::PerHostBase humanBase;
    
    
    /** Duration of host-seeking per day; the maximum fraction of a day that a
     * mosquito would spend seeking (θ_d). */
    double mosqSeekingDuration;
    
    
    /** @brief Probabilities and rates associated with life-cycle model
     * 
     * These are calculated during initialisation and thereafter constant.
     * 
     * Probabilities have no units; others have units specified.
     *
     * All parameters are calculated during initialisation and in theory don't
     * need checkpointing. */
    //@{
    /** Death rate of mosquitoes while host-seeking (μ_vA).
     *
     * Unit: animals/day. */
    double mosqSeekingDeathRate;

    /** Probability of a mosquito successfully laying eggs given that it has
     * rested (P_E).
     *
     * Currently assumed constant, although NC's non-autonomous model provides
     * an alternative. */
    double probMosqSurvivalOvipositing;

    struct NHHParams {
        // N_i * α_i: rate at which mosquitoes encouter hosts of this type
        double entoAvailability;
        // α_i * P_B_i * P_C_i * P_D_i
        double probCompleteCycle;
    };
    /** Non-human host data. Doesn't need checkpointing. */
    vector<NHHParams> nonHumans;
    //@}
    
    struct TrapParams {
        // Initial availability of a trap relative to an adult
        double relAvail;
        // Decay of availability
        // note: can't use "auto_ptr" inside a vector, otherwise that would suffice
        boost::shared_ptr<util::DecayFunction> availDecay;
    };
    // Parameters for trap interventions. Doesn't need checkpointing.
    vector<TrapParams> trapParams;
    
    
    // -----  model state (and some encapsulated parameters)  -----
    
    /** @brief transmission and life-cycle parts of model
     * 
     * Much of the core model is encapsulated here. */
    MosqTransmission transmission;
    
    /** @brief Intervention parameters */
    //@{
    /** Interventions affecting death rate while seeking (parameters + state)
     * 
     * Value is increase in death rate (so multiply rate by 1 + this). */
    vector<util::SimpleDecayingValue> seekingDeathRateIntervs;
    /** Interventions affecting probability of dying while ovipositing (pre laying of eggs) (parameters + state)
     * 
     * Value is probability of dying due to this intervention (so multiply survival by 1 - this). */
    vector<util::SimpleDecayingValue> probDeathOvipositingIntervs;
    struct TrapData {
        // index in trapParams
        size_t instance;
        // initial availability (avail per trap * num traps)
        double initialAvail;
        // parameter for decay of availability
        DecayFuncHet availHet;
        // deploy time (for decay function)
        SimTime deployTime;
        // date at which this intervention should be deleted
        SimTime expiry;
        
        /// Checkpointing
        template<class S>
        void operator& (S& stream) {
            instance & stream;
            initialAvail & stream;
            availHet & stream;
            deployTime & stream;
            expiry & stream;
        }
    };
    /** Baited trap interventions.
     * 
     * This is the "full" availability: availability per trap times the number
     * of traps. Each deployment has its own availiability along with expiry
     * date; total availability is the sum. */
    list<TrapData> baitedTraps;
    //@}
    
    /** Per time-step partial calculation of EIR, per genotype.
    *
    * See comment in advancePeriod() for details of how the EIR is calculated.
    *
    * Doesn't need to be checkpointed (is recalculated each step). */
    vector<double> partialEIR;
};

}
}
}
#endif
