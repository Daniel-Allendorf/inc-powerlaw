/*******************************************************************************
 * sampling/hypergeometric_distribution.hpp
 *
 * This code was taken from
 * https://github.com/lorenzhs/sampling/blob/master/sampling/hypergeometric_distribution.hpp
 * and slightly modified (template random source).
 *
 * A hypergeomitric distribution random generator adapted from NumPy.
 *
 * The implementation of loggam(), rk_hypergeometric(), rk_hypergeometric_hyp(),
 * and rk_hypergeometric_hrua() were adapted from NumPy's
 * numpy/random/mtrand/distributions.c which has this license:
 *
 * Copyright 2005 Robert Kern (robert.kern@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * The implementations of rk_hypergeometric_hyp(), rk_hypergeometric_hrua(),
 * and rk_triangular() were adapted from Ivan Frohne's rv.py which has this
 * license:
 *
 *            Copyright 1998 by Ivan Frohne; Wasilla, Alaska, U.S.A.
 *                            All Rights Reserved
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation for any purpose, free of charge, is granted subject to the
 * following conditions:
 *   The above copyright notice and this permission notice shall be included in
 *   all copies or substantial portions of the software.
 *
 *   THE SOFTWARE AND DOCUMENTATION IS PROVIDED WITHOUT WARRANTY OF ANY KIND,
 *   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO MERCHANTABILITY, FITNESS
 *   FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHOR
 *   OR COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM OR DAMAGES IN A CONTRACT
 *   ACTION, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 *   SOFTWARE OR ITS DOCUMENTATION.
 *
 *
 * Copyright (C) 2017 Lorenz Hübschle-Schneider <lorenz@4z2.de>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <cassert>
#include <array>
#include <cmath>
#include <random>
#include <tlx/define.hpp>

namespace shuffle {

template <typename PRNG>
class HypergeometricDistribution
{
public:
    HypergeometricDistribution(PRNG& gen) : gen_(gen) {}

    int64_t operator()(int64_t good, int64_t bad, int64_t sample) {
        assert(sample > 0 && sample < good + bad);
        return rk_hypergeometric(good, bad, sample);
    }

    size_t variates_obtained() const {
        return variates_obtained_;
    }

    size_t loops() const {
        return loops_;
    }


private:
    static constexpr std::array<double, 126> precomputed_logfac {
        0,
        0,
        0.69314718055994529,
        1.791759469228055,
        3.1780538303479458,
        4.7874917427820458,
        6.5792512120101012,
        8.5251613610654147,
        10.604602902745251,
        12.801827480081469,
        15.104412573075516,
        17.502307845873887,
        19.987214495661885,
        22.552163853123425,
        25.19122118273868,
        27.89927138384089,
        30.671860106080672,
        33.505073450136891,
        36.395445208033053,
        39.339884187199495,
        42.335616460753485,
        45.380138898476908,
        48.471181351835227,
        51.606675567764377,
        54.784729398112319,
        58.003605222980518,
        61.261701761002001,
        64.557538627006338,
        67.88974313718154,
        71.257038967168015,
        74.658236348830158,
        78.092223553315307,
        81.557959456115043,
        85.054467017581516,
        88.580827542197682,
        92.136175603687093,
        95.719694542143202,
        99.330612454787428,
        102.96819861451381,
        106.63176026064346,
        110.32063971475739,
        114.03421178146171,
        117.77188139974507,
        121.53308151543864,
        125.3172711493569,
        129.12393363912722,
        132.95257503561632,
        136.80272263732635,
        140.67392364823425,
        144.5657439463449,
        148.47776695177302,
        152.40959258449735,
        156.3608363030788,
        160.3311282166309,
        164.32011226319517,
        168.32744544842765,
        172.35279713916279,
        176.39584840699735,
        180.45629141754378,
        184.53382886144948,
        188.6281734236716,
        192.7390472878449,
        196.86618167289001,
        201.00931639928152,
        205.1681994826412,
        209.34258675253685,
        213.53224149456327,
        217.73693411395422,
        221.95644181913033,
        226.1905483237276,
        230.43904356577696,
        234.70172344281826,
        238.97838956183432,
        243.26884900298271,
        247.57291409618688,
        251.89040220972319,
        256.22113555000954,
        260.56494097186322,
        264.92164979855278,
        269.29109765101981,
        273.67312428569369,
        278.06757344036612,
        282.4742926876304,
        286.89313329542699,
        291.32395009427029,
        295.76660135076065,
        300.22094864701415,
        304.68685676566872,
        309.1641935801469,
        313.65282994987905,
        318.1526396202093,
        322.66349912672615,
        327.1852877037752,
        331.71788719692847,
        336.26118197919845,
        340.81505887079902,
        345.37940706226686,
        349.95411804077025,
        354.53908551944079,
        359.1342053695754,
        363.73937555556347,
        368.35449607240474,
        372.97946888568902,
        377.61419787391867,
        382.25858877306001,
        386.91254912321756,
        391.57598821732961,
        396.24881705179155,
        400.93094827891576,
        405.6222961611449,
        410.32277652693733,
        415.03230672824964,
        419.75080559954472,
        424.47819341825709,
        429.21439186665157,
        433.95932399501481,
        438.71291418612117,
        443.47508812091894,
        448.24577274538461,
        453.02489623849613,
        457.81238798127816,
        462.60817852687489,
        467.4121995716082,
        472.22438392698058,
        477.04466549258564,
        481.87297922988796
    };

    constexpr static double halfln2pi = 0.9189385332046728;
    constexpr static double D1 = 1.7155277699214135; // 2*sqrt(2/e)
    constexpr static double D2 = 0.8989161620588988; // 3 - 2*sqrt(3/e)

    double logfactorial(int64_t k) const
    {
        assert(k >= 0);

        if (static_cast<size_t>(k) < precomputed_logfac.size()) {
            /* Use the lookup table. */
            return precomputed_logfac[k];
        }

        /*
         *  Use the Stirling series, truncated at the 1/k**3 term.
         *  (In a Python implementation of this approximation, the result
         *  was within 2 ULP of the best 64 bit floating point value for
         *  k up to 10000000.)
         */
        return (k + 0.5)*std::log(k) - k + (halfln2pi + (1.0/k)*(1/12.0 - 1/(360.0*k*k)));
    }

    int64_t hypergeometric_hrua(int64_t good, int64_t bad, int64_t sample)
    {
        int64_t mingoodbad, maxgoodbad, popsize;
        int64_t computed_sample;
        double p, q;
        double mu, var;
        double a, c, b, h, g;
        int64_t m, K;

        popsize = good + bad;
        computed_sample = std::min(sample, popsize - sample);
        mingoodbad = std::min(good, bad);
        maxgoodbad = std::max(good, bad);

        /*
         *  Variables that do not match Stadlober (1989)
         *    Here               Stadlober
         *    ----------------   ---------
         *    mingoodbad            M
         *    popsize               N
         *    computed_sample       n
         */

        p = ((double) mingoodbad) / popsize;
        q = ((double) maxgoodbad) / popsize;

        // mu is the mean of the distribution.
        mu = computed_sample * p;

        a = mu + 0.5;

        // var is the variance of the distribution.
        var = ((double)(popsize - computed_sample) *
               computed_sample * p * q / (popsize - 1));

        c = sqrt(var + 0.5);

        /*
         *  h is 2*s_hat (See Stadlober's theses (1989), Eq. (5.17); or
         *  Stadlober (1990), Eq. 8).  s_hat is the scale of the "table mountain"
         *  function that dominates the scaled hypergeometric PMF ("scaled" means
         *  normalized to have a maximum value of 1).
         */
        h = D1*c + D2;

        m = (int64_t) floor((double)(computed_sample + 1) * (mingoodbad + 1) /
                            (popsize + 2));

        g = (logfactorial(m) +
             logfactorial(mingoodbad - m) +
             logfactorial(computed_sample - m) +
             logfactorial(maxgoodbad - computed_sample + m));

        /*
         *  b is the upper bound for random samples:
         *  ... min(computed_sample, mingoodbad) + 1 is the length of the support.
         *  ... floor(a + 16*c) is 16 standard deviations beyond the mean.
         *
         *  The idea behind the second upper bound is that values that far out in
         *  the tail have negligible probabilities.
         *
         *  There is a comment in a previous version of this algorithm that says
         *      "16 for 16-decimal-digit precision in D1 and D2",
         *  but there is no documented justification for this value.  A lower value
         *  might work just as well, but I've kept the value 16 here.
         */
        b = std::min<int64_t>(std::min(computed_sample, mingoodbad) + 1, floor(a + 16*c));

        while (1) {
            const auto U = uniform_();
            const auto V = uniform_();  // "U star" in Stadlober (1989)
            const auto X = a + h*(V - 0.5) / U;

            // fast rejection:
            if ((X < 0.0) || (X >= b)) {
                continue;
            }

            K = (int64_t) floor(X);

            const auto T = g - (logfactorial(K) +
                  logfactorial(mingoodbad - K) +
                  logfactorial(computed_sample - K) +
                  logfactorial(maxgoodbad - computed_sample + K));


            // fast acceptance:
            if ((U*(4.0 - U) - 3.0) <= T) {
                break;
            }

            // fast rejection:
            if (U*(U - T) >= 1) {
                continue;
            }

            if (2.0*log(U) <= T) {
                // acceptance
                break;
            }
        }

        if (good > bad) {
            K = computed_sample - K;
        }

        if (computed_sample < sample) {
            K = good - K;
        }

        return K;
    }

    int64_t rk_hypergeometric_int(int64_t good, int64_t bad, int64_t sample) {
        if (bad < good)
            return sample - rk_hypergeometric_int(bad, good, sample);

        static_assert(gen_.min() == 0, "RNG not supported");
        static_assert(gen_.max() >= std::numeric_limits<int64_t>::max(), "RNG not supported");

        auto total = good + bad;
        const auto scaler = std::numeric_limits<int64_t>::max() / total;
        auto scaled_total = total * scaler;
        auto scaled_remaining_good = good * scaler;

        int64_t taken = 0;
        auto inner_loop = [&] () {
            while(1) {
                // obtain a uniform integer < scaled_total (using rejection sampling)
                // observe that scaled_total is near int-max and thus rejection is unlikely
                const auto unif = static_cast<int64_t>(gen_());
                if (TLX_UNLIKELY(unif >= scaled_total))
                    continue; // reject

                const auto take_good = (unif < scaled_remaining_good);
                scaled_remaining_good -= take_good * scaler;
                taken += take_good;
                scaled_total -= scaler;

                return;
            }
        };

        for (auto i = sample; i; --i) {
            inner_loop();
            if (!TLX_UNLIKELY(scaled_remaining_good)) return taken;
        }

        return taken;
    }

    int64_t rk_hypergeometric(int64_t good, int64_t bad, int64_t sample) {
        variates_obtained_++;
        if (sample < 10) {
            return rk_hypergeometric_int(good, bad, sample);
        }

        loops_++;
        return hypergeometric_hrua(good, bad, sample);
    }

    // Data members:
    PRNG& gen_;
    std::uniform_real_distribution<double> real_;
    size_t variates_obtained_{0};
    size_t loops_{0};

    double uniform_() {return real_(gen_);}
};

} // namespace sampling

