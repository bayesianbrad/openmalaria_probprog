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

/* This module contains the random-number generator and distributions wrapper.
 *
 * Currently both the GSL and boost generators are implemented. The
 * distributions all come from the GSL library so far.
 *
 * Using the boost generator appears (in rough tests) to be slightly
 * slower, which is understandable since the GSL distributions must then use a
 * wrapper around the boost generator.
 *
 * Note: using boost distributions elsewhere could ideally be implemented a
 * little differently, since the distribution objects could in many cases last
 * the length of the program rather than be created on each use.
 */

// Define to use boost as the underlying generator:
//#define OM_RANDOM_USE_BOOST

#include "util/random.h"
#include "util/errors.h"
#include "util/StreamValidator.h"
#include "Global.h"
#ifdef OM_RANDOM_USE_BOOST
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/lognormal_distribution.hpp>
#include <boost/static_assert.hpp>
#endif
#include <iostream>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include <cmath>
#include <sstream>
#include <pyprob_cpp.h>
#include "xtensor/xadapt.hpp"
// Note: since we're using both gsl and boost files, we should be careful to
// avoid name conflicts. So probably don't use "using namespace boost;".


namespace OM { namespace util {
# ifdef OM_RANDOM_USE_BOOST
    static boost::mt19937 boost_generator;
    static boost::uniform_01<boost::mt19937&> rng_uniform01 (boost_generator);

    long unsigned int boost_rng_get (void*) {
    BOOST_STATIC_ASSERT (sizeof(uint32_t) <= sizeof(long unsigned int));
    long unsigned int val = static_cast<long unsigned int> (boost_generator ());
    streamValidate( val );
    return val;
    }
    double boost_rng_get_double_01 (void*) {
    return rng_uniform01 ();
    }

    static const gsl_rng_type boost_mt_type = {
    "boost_mt19937",        // name
    boost_generator.max(),  // max value
    boost_generator.min(),  // min value
    0,                  // size of state; not used here
    NULL,               // re-seed function; don't use
    &boost_rng_get,
    &boost_rng_get_double_01
    };
# endif

// This should be created and deleted automatically, taking care of
// allocating and freeing the generator.
struct generator_factory {
    gsl_rng * gsl_generator;

    generator_factory () {
#   ifdef OM_RANDOM_USE_BOOST
    // In this case, I construct a wrapper around boost's generator. The reason for this is
    // that it allows use of distributions from both boost and GSL.
    gsl_generator = new gsl_rng;
    gsl_generator->type = &boost_mt_type;
    gsl_generator->state = NULL;    // state is stored as static variables
#   else
    //use the mersenne twister generator
    gsl_generator = gsl_rng_alloc(gsl_rng_mt19937);
#   endif
    }
    ~generator_factory () {
#   ifdef OM_RANDOM_USE_BOOST
    delete gsl_generator;
#   else
    gsl_rng_free (gsl_generator);
#   endif
    }
} rng;

// -----  set-up, tear-down and checkpointing  -----

void random::seed (uint32_t seed) {
    printf("random::seed\n");

//     util::streamValidate(seed);
# ifdef OM_RANDOM_USE_BOOST
    if (seed == 0) seed = 4357; // gsl compatibility − ugh
    boost_generator.seed (seed);
# else
    gsl_rng_set (rng.gsl_generator, seed);
# endif
}

void random::checkpoint (istream& stream, int seedFileNumber) {
    printf("random::checkpoint\n");

# ifdef OM_RANDOM_USE_BOOST
    // Don't use OM::util::checkpoint function for loading a stream; checkpoint::validateListSize uses too small a number.
    string str;
    size_t len;
    len & stream;
    str.resize (len);
    stream.read (&str[0], str.length());
    if (!stream || stream.gcount() != streamsize(len))
    throw checkpoint_error ("stream read error string");
    istringstream ss (str);
    ss >> boost_generator;
# else

    ostringstream seedN;
    seedN << string("seed") << seedFileNumber;
    FILE * f = fopen(seedN.str().c_str(), "rb");
    if (f == NULL)
    throw checkpoint_error (string("load_rng_state: file not found: ").append(seedN.str()));
    if (gsl_rng_fread(f, rng.gsl_generator) != 0)
    throw checkpoint_error ("gsl_rng_fread failed");
    fclose (f);
# endif
}

void random::checkpoint (ostream& stream, int seedFileNumber) {
    printf("random::checkpoint\n");

# ifdef OM_RANDOM_USE_BOOST
    ostringstream ss;
    ss << boost_generator;
    ss.str() & stream;
# else

    ostringstream seedN;
    seedN << string("seed") << seedFileNumber;
    FILE * f = fopen(seedN.str().c_str(), "wb");
    if (gsl_rng_fwrite(f, rng.gsl_generator) != 0)
    throw checkpoint_error ("gsl_rng_fwrite failed");
    fclose (f);
# endif
}


// -----  random number generation  -----

double random::uniform_01 () {
    // printf("Pyprob uniform 0 1\n");
    auto uniform = pyprob_cpp::distributions::Uniform(0,1);
    // cout<<" Unifrom 0 1 sample "<<sample<<endl;
    return pyprob_cpp::sample(uniform)(0);
    // return pyprob_cpp::sample(uniform)(0); // this returns a tensor which is a single element

// double result =
//     // GSL and boost versions both do the same (when using boost as the underlying generator):
// # ifdef OM_RANDOM_USE_BOOST
//         rng_uniform01 ();
// # else
//         gsl_rng_uniform (rng.gsl_generator);
// # endif
// //     util::streamValidate(result);
//     return result;
}

double random::gauss (double mean, double std){
    // printf("Pyprob normal mean std \n");
    auto normal = pyprob_cpp::distributions::Normal(mean, std);

    // double result = gsl_ran_gaussian(rng.gsl_generator,std)+mean;
    // util::streamValidate(result);
    // return result;
    return pyprob_cpp::sample(normal)(0);
}
// double random::gauss (double std){
//     printf("random::gauss\n");

//     double result = gsl_ran_gaussian(rng.gsl_generator,std);
// //     util::streamValidate(result);
//     return result;
// }

double random::gamma (double a, double b){
    printf("random::gamma\n");

    // double result = gsl_ran_gamma(rng.gsl_generator, a, b);
//     util::streamValidate(result);
    // return result;
    auto gamma = pyprob_cpp::distributions::Gamma(a, b);
    return pyprob_cpp::sample(gamma)(0);
}

// double random::log_normal (double mu, double sigma){
//     printf("random::log_normal\n");

// /*# ifdef OM_RANDOM_USE_BOOST
//     // This doesn't work: boost version takes mean and sigma while gsl version takes mu and sigma.
//     boost::lognormal_distribution<> dist (mean, std);
//     return dist (boost_generator);
// # else*/
// //     double result = gsl_ran_lognormal (rng.gsl_generator, mu, sigma);
// // //     util::streamValidate(result);
// //     return result;
// //# endif
//     auto log_normal = pyprob_cpp::distributions::LogNormal(mu, sigma);
//      cout << "Bebug statemennt in log_normal random.cpp : \n" << pyprob_cpp::sample(log_normal)(0);
//     return pyprob_cpp::sample(log_normal)(0);
// }

// double random::sampleFromLogNormal(double normp, double meanlog, double stdlog){
//     printf("random::sample_log_normal\n");
//     // Used for performance reasons. Calling GSL's log_normal 5 times is 50% slower.
//     // printf("random::sampleFromLogNormal\n");

//     double zval = gsl_cdf_ugaussian_Pinv (normp);
// //     util::streamValidate(zval);
//     // Where normp is distributed uniformly over [0,1], this acts like a sample
//     // from the log normal. Where normp has been transformed by raising the
//     // uniform sample to the power of 1/(T-1), zval is distributed like a
//     // uniform gauss times 4* F(x,0,1)^3, where F(x,0,1) ist the cummulative
//     // distr. function of a uniform gauss:
//     // double result = exp(meanlog+stdlog*zval);
//     auto log_normal = pyprob_cpp::distributions::LogNormal(exp(meanlog), exp(stdlog));
//     cout << "Bebug statemennt in LOG_normal random.cpp : \n" << pyprob_cpp::sample(log_normal)(0) ;
//     return pyprob_cpp::sample(log_normal)(0);
// //     util::streamValidate(result);
//     // return result;
// }

double random::log_normal (double mu, double sigma){
/*# ifdef OM_RANDOM_USE_BOOST
    // This doesn't work: boost version takes mean and sigma while gsl version takes mu and sigma.
    boost::lognormal_distribution<> dist (mean, std);
    return dist (boost_generator);
# else*/
    double result = gsl_ran_lognormal (rng.gsl_generator, mu, sigma);
//     util::streamValidate(result);
    return result;
//# endif
}

double random::sampleFromLogNormal(double normp, double meanlog, double stdlog){
    // Used for performance reasons. Calling GSL's log_normal 5 times is 50% slower.
    
    double zval = gsl_cdf_ugaussian_Pinv (normp);
//     util::streamValidate(zval);
    // Where normp is distributed uniformly over [0,1], this acts like a sample
    // from the log normal. Where normp has been transformed by raising the
    // uniform sample to the power of 1/(T-1), zval is distributed like a
    // uniform gauss times 4* F(x,0,1)^3, where F(x,0,1) ist the cummulative
    // distr. function of a uniform gauss:
    double result = exp(meanlog+stdlog*zval);
//     util::streamValidate(result);
    return result;
}
double random::beta (double a, double b){
    // printf("random::beta\n");

    double result = gsl_ran_beta (rng.gsl_generator,a,b);
//     util::streamValidate(result);
    return result;
}
double random::betaWithMean (double m, double b){
    // printf("random::betaWithMean\n");

    //TODO(performance): could do this calculation externally, and feed in a,b instead of mean,b
    double a = m * b / (1.0 - m);
//     util::streamValidate(a);
    return beta(a,b);
}

int random::poisson(double lambda){
  
    if( !(boost::math::isfinite)(lambda) ){
    //This would lead to an inifinite loop in gsl_ran_poisson
    throw TRACED_EXCEPTION( "lambda is inf", Error::InfLambda );
    }
    // printf("Pyprob possion 1\n");
    auto poisson = pyprob_cpp::distributions::Poisson(lambda);

    // for debugging 
    // auto sample = pyprob_cpp::sample(poisson)(0);
    // cout<< "poisson sample: "<<sample<<endl;
    return pyprob_cpp::sample(poisson)(0);
    // int result = gsl_ran_poisson (rng.gsl_generator, lambda);
//     util::streamValidate(result);
    // return result;
}

bool random::bernoulli(double prob){
    // printf("random::bernoulli\n");

    assert( (boost::math::isfinite)(prob) );
    // return true iff our variate is less than the probability
    bool result =random::uniform_01() < prob;
//     util::streamValidate(result);
    return result;
}

int random::uniform(int n){
    // printf("random::uniform\n");
    assert( (boost::math::isfinite)(n) );
    return static_cast<int>( random::uniform_01() * n );
}

double random::exponential(double mean){
    // printf("random::exponential\n");
    // return gsl_ran_exponential(rng.gsl_generator, mean);
    auto exponential = pyprob_cpp::distributions::Exponential(mean);
    return pyprob_cpp::sample(exponential)(0);
}

double random::weibull(double lambda, double k){
    // printf("random::weibull\n");
    // return gsl_ran_weibull( rng.gsl_generator, lambda, k );
    auto weibull = pyprob_cpp::distributions::Weibull(lambda, k);
    return pyprob_cpp::sample(weibull)(0);
}

} }


