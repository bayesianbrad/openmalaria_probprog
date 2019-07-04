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

#ifndef Hmod_LSTMDrug
#define Hmod_LSTMDrug

#include "Global.h"
#include "util/checkpoint_containers.h"

namespace OM {
namespace PkPd {

/** A class holding pkpd drug use info.
 *
 * Each human has an instance for each type of drug present in their blood. */
class LSTMDrug {
public:
    /// Create a new instance.
    /// Volume of distribution must be specified here (from sample or mean).
    LSTMDrug (double Vd);
    /// Obligatory virtual destructor on a virtual class
    virtual ~LSTMDrug();
    
    /// Get the drug type's index.
    /// TODO: decide whether this should be virtual or the index should be a local
    virtual size_t getIndex() const =0;
    
    /** Indicate a new medication this time step.
     *
     * Converts qty in mg to concentration, and stores along with time (delay past
     * the start of the current time step) in the doses container.
     * 
     * @param time Time of administration, in days (should be at least 0 and
     *  less than 1).
     * @param qty Amount of active ingredient, in mg
     * @param bodyMass Body mass of patient, in kg
     */
    virtual void medicate (double time, double qty, double bodyMass) =0;
    
    /** Get the concentration of the given drug contained in this model (only
     * compartments with active PD; zero if drug index doesn't match that used).
     * 
     * @returns Concentration in the blood serum, in mg/l. */
    virtual double getConcentration(size_t index) const =0;
    
    /** Returns the total drug factor for one drug over one day.
     *
     * The drug factor values generated by this function must be multiplied to
     * reflect the drug action of all drugs in one day.
     *
     * This doesn't adjust concentration because this function may be called
     * several times (for each infection) per time step, or not at all.
     * 
     * @param genotype An identifier for the genotype of the infection.
     * @param body_mass Weight of patient in kg
     */
    virtual double calculateDrugFactor(uint32_t genotype, double body_mass) const =0;
    
    /** Updates concentration variable and clears day's doses.
     * 
     * @param body_mass Weight of patient in kg */
    virtual void updateConcentration (double body_mass) =0;
    
    /// Checkpointing
    template<class S>
    void operator& (S& stream) {
        doses & stream;
        checkpoint(stream);
    }

protected:
    /** Indicate a new medication this time step, specifying volume of
     * distribution directly.
     *
     * Converts qty in mg to concentration, and stores along with time (delay past
     * the start of the current time step) in the doses container.
     * 
     * @param time Time of administration, in days (should be at least 0 and
     *  less than 1).
     * @param qty Amount of active ingredient, in mg
     * @param volDist Volume of distribution in l
     */
    void medicate_vd (double time, double qty, double volDist);
    
    virtual void checkpoint (istream& stream){}
    virtual void checkpoint (ostream& stream){}
    
    /// First is time (days), second is additional concentration (mg / l; for
    /// one- and three-compartment models) or quantity (mg; for conversion model)
    typedef std::vector<std::pair<double,double> > DoseVec;
    /** List of each dose given today (and possibly tomorrow), ordered by time.
     * First parameter (key) is time in days, second is the dose concentration (mg/l).
     *
     * Read in calculateDrugFactor, and updated in updateConcentration(). */
    DoseVec doses;
    
    /// Volume of distribution, sampled when this class is first created.
    double vol_dist;
};

}
}
#endif
