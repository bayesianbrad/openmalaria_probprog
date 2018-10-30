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

#ifndef Hmod_AnophelesLifeCycle
#define Hmod_AnophelesLifeCycle

#include "Global.h"
#include <vector>

namespace scnXml {
    class LifeCycle;
}

class MosqLifeCycleSuite;

namespace OM {
namespace Transmission {
namespace Anopheles {

class LifeCycle;

/** Parameters for the mosquito life cycle (population dynamics) model.
 * 
 * Chitnis: “A Periodically-Forced Difference Equation Model for Mosquito
 * Population Dynamics” (17th June 2011, unpublished).
 */
class LifeCycleParams {
public:
    /** Initialises mosquito life-cycle parameters. */
    void initLifeCycle( const scnXml::LifeCycle& lifeCycle );
    
    /** Get larval resources available during the last time-step. Intended for
     * reporting; not especially fast. */
    double getResAvailability() const;
    inline int getTotalDuration() const{
        return eggStageDuration + larvalStageDuration + pupalStageDuration;
    }
    
    /// Checkpointing
    template<class S>
    void operator& (S& stream) {
        invLarvalResources & stream;
    }
    
    /** Fit larvalResources from S_v (which is derived from EIR).
     * 
     * @param lcModel LifeCycle state to start from
     * @param P_A Average P_A value (assumed constant)
     * @param P_df Average P_df value (assumed constant)
     * @param N_v_length Parameter from AnophelesModel
     * @param mosqRestDuration The duration of a feeding cycle (τ)
     */
    void fitLarvalResourcesFromS_v(
        const LifeCycle& lcModel,
        double P_A, double P_df,
        SimTime N_v_length, SimTime mosqRestDuration,
        vector<double>& annualP_dif,
        vector<double>& targetS_v
    );
    
private:
    /** @brief Duration parameters for mosquito/parasite life-cycle
     * 
     * Currently these are all constant. In theory they could be made to vary
     * seasonally, based on a fixed periodic cycle, though some code and
     * possibly model changes would be needed to accomodate this.
     * 
     * All have units of days.
     *
     * Set in initialise function from XML data; no need to checkpoint. */
    //@{
    /** Duration of egg stage (time from laying until hatching) (θ_e).
     * Units: days. */
    SimTime eggStageDuration;

    /** Duration of larval stage (time from hatching until becoming a pupa)
     * (θ_l). Units: days. */
    SimTime larvalStageDuration;

    /** Duration of pupal stage (time from becoming a pupa until emerging as an
     * adult) (θ_p). Units: days. */
    SimTime pupalStageDuration;
    //@}
    
    /** @brief Mosquito population-dynamics parameters
     * 
     * Probabilities have no units; others have units specified.
     *
     * All parameters are calculated during initialisation and in theory don't
     * need checkpointing. */
    //@{
    /** Probability of an egg which has been laid hatching (ρ_e ^ θ_e). */
    double pSurvEggStage;
    
    /** Probability of a larva surviving one day, assuming no resource
     * restrictions (ρ_l). */
    double pSurvDayAsLarvae;
    
    /** Probability of a new pupa emerging as an adult (ρ_p ^ θ_p). */
    double pSurvPupalStage;
    
    /** Mean number of female eggs laid when a mosquito oviposites. */
    double fEggsLaidByOviposit;
    
    /** Initial larval resources guess used when fitting. */
    double estimatedLarvalResources;
    
    /** Resource usage of female larvae by age.
     * 
     * Length: θ_l. Index i corresponds to usage at age i days after hatching.
     * 
     * Units: usage/larva. Units of usage are not defined, but should be the
     * same as that of resource availability. */
    vecDay<double> larvaeResourceUsage;
    
    /** @brief Measure of larval resources (1/γ)
     * Inverse of resource availability to female larvae throughout the year.
     * Note that since male larvae are not modelled, the proportion of
     * resources used by males should not be included here.
     * 
     * Has annual periodicity: length is 365. First value (index 0) corresponds
     * to first day of year (1st Jan or something else if rebased). In 5-day
     * time-step model values at indecies 0 through 4 are used to calculate the
     * state at time-step 1.
     *
     * Units: not defined, but must match the unit of resource usage.
     * 
     * Note: this parameter needs to be checkpointed since it is calculated
     * during init. */
    vecDay<double> invLarvalResources;
    
    /** Effect of competition on larvae, per age (index i corresponds to age i
     * days since hatching).
     * 
     * Length: larvalStageDuration */
    vecDay<double> effectCompetitionOnLarvae;
    //@}
    friend class LifeCycle;
    friend class ResourceFitter;
    friend class ::MosqLifeCycleSuite;
};


/** Encapsulates state of mosquito life cycle (population dynamics) model.
 * 
 * Chitnis: “A Periodically-Forced Difference Equation Model for Mosquito
 * Population Dynamics” (17th June 2011, unpublished).
 */
class LifeCycle {
public:
    /** Initialise/reset state variables to 0.
     * 
     * Note that output of updateEmergence shouldn't be used before
     * lcParams.getTotalDuration() updates have occurred after initialisation
     * or reset.
     * 
     * @param lcParams Fixed parameters for the life-cycle model
     */
    void init( const LifeCycleParams& lcParams );
    
    /** Return the theoretical resource requirements of this vector at this
     * time-step (note that, due to Beverton–Holt model used, some growth
     * restriction still occurs with this level of resouce availability). */
    double getResRequirements( const LifeCycleParams& lcParams ) const;
    
    /** Update state and return the number of newly emerging (mated) female
     * adults.
     * 
     * @param lcParams Fixed parameters for the life-cycle model
     * @param nOvipositingMosqs The number of adults which successfully
     * oviposited this/last time-step. TODO(vec lifecycle): we're setting new
     * value based on num ovipositing yesterday? That's not right.
     * @param d0 Time of the start of the update period
     * @returns The number of adults emerging between the last simulated time
     * point and the one being calculated. Assume immediate mating with 100%
     * survival and success.
     */
    double updateEmergence( const LifeCycleParams& lcParams,
                            double nOvipositingMosqs,
                            SimTime d0 );
    
    /// Checkpointing
    template<class S>
    void operator& (S& stream) {
        newEggs & stream;
        numLarvae & stream;
        newPupae & stream;
    }
    
private:
    /** Number of eggs laid per time-step (ϒ_e). Units: eggs.
     * 
     * Length: θ_e. Value at index (d mod θ_e) refers to the value θ_e days
     * ago/at day d before/after update. */
    vecDay<double> newEggs;
    
    /** Number of larvae per age of development. Units: larvae.
     * 
     * Length: θ_l. Value at index i refers to the number of larvae of age i.
     * We don't store the number at age θ_l, since these are pupae.
     *
     * Unlike ϒ arrays, this only stores the state of the system from the
     * last/this time step before/after update. */
    vecDay<double> numLarvae;
    
    /** Number of new pupae per time-step (ϒ_e). Units: pupae.
     * 
     * Length: θ_p. Value at index (d mod θ_p) refers to the value θ_p days
     * ago/at day d before/after update. */
    vecDay<double> newPupae;
    
    friend class ::MosqLifeCycleSuite;
    friend class ResourceFitter;        // TODO(vec lifecycle): this is temporary for debugging
};

}
}
}

#endif
