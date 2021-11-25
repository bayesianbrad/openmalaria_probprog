/* Generated file, do not edit */

#ifndef CXXTEST_RUNNING
#define CXXTEST_RUNNING
#endif

#define _CXXTEST_HAVE_STD
#define _CXXTEST_HAVE_EH
#define _CXXTEST_ABORT_TEST_ON_FAIL
#include <cxxtest/TestListener.h>
#include <cxxtest/TestTracker.h>
#include <cxxtest/TestRunner.h>
#include <cxxtest/RealDescriptions.h>
#include <cxxtest/ParenPrinter.h>

int main() {
 return CxxTest::ParenPrinter().run();
}
#include "ExtraAsserts.h"

static ExtraAssertsSuite suite_ExtraAssertsSuite;

static CxxTest::List Tests_ExtraAssertsSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_ExtraAssertsSuite( "ExtraAsserts.h", 179, "ExtraAssertsSuite", suite_ExtraAssertsSuite, Tests_ExtraAssertsSuite );

static class TestDescription_ExtraAssertsSuite_testIEEE754 : public CxxTest::RealTestDescription {
public:
 TestDescription_ExtraAssertsSuite_testIEEE754() : CxxTest::RealTestDescription( Tests_ExtraAssertsSuite, suiteDescription_ExtraAssertsSuite, 187, "testIEEE754" ) {}
 void runTest() { suite_ExtraAssertsSuite.testIEEE754(); }
} testDescription_ExtraAssertsSuite_testIEEE754;

static class TestDescription_ExtraAssertsSuite_testApproxEq : public CxxTest::RealTestDescription {
public:
 TestDescription_ExtraAssertsSuite_testApproxEq() : CxxTest::RealTestDescription( Tests_ExtraAssertsSuite, suiteDescription_ExtraAssertsSuite, 200, "testApproxEq" ) {}
 void runTest() { suite_ExtraAssertsSuite.testApproxEq(); }
} testDescription_ExtraAssertsSuite_testApproxEq;

#include "LSTMPkPdSuite.h"

static LSTMPkPdSuite suite_LSTMPkPdSuite;

static CxxTest::List Tests_LSTMPkPdSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_LSTMPkPdSuite( "LSTMPkPdSuite.h", 37, "LSTMPkPdSuite", suite_LSTMPkPdSuite, Tests_LSTMPkPdSuite );

static class TestDescription_LSTMPkPdSuite_testNone : public CxxTest::RealTestDescription {
public:
 TestDescription_LSTMPkPdSuite_testNone() : CxxTest::RealTestDescription( Tests_LSTMPkPdSuite, suiteDescription_LSTMPkPdSuite, 61, "testNone" ) {}
 void runTest() { suite_LSTMPkPdSuite.testNone(); }
} testDescription_LSTMPkPdSuite_testNone;

static class TestDescription_LSTMPkPdSuite_testOral : public CxxTest::RealTestDescription {
public:
 TestDescription_LSTMPkPdSuite_testOral() : CxxTest::RealTestDescription( Tests_LSTMPkPdSuite, suiteDescription_LSTMPkPdSuite, 65, "testOral" ) {}
 void runTest() { suite_LSTMPkPdSuite.testOral(); }
} testDescription_LSTMPkPdSuite_testOral;

static class TestDescription_LSTMPkPdSuite_testOralHalves : public CxxTest::RealTestDescription {
public:
 TestDescription_LSTMPkPdSuite_testOralHalves() : CxxTest::RealTestDescription( Tests_LSTMPkPdSuite, suiteDescription_LSTMPkPdSuite, 70, "testOralHalves" ) {}
 void runTest() { suite_LSTMPkPdSuite.testOralHalves(); }
} testDescription_LSTMPkPdSuite_testOralHalves;

static class TestDescription_LSTMPkPdSuite_testOralSplit : public CxxTest::RealTestDescription {
public:
 TestDescription_LSTMPkPdSuite_testOralSplit() : CxxTest::RealTestDescription( Tests_LSTMPkPdSuite, suiteDescription_LSTMPkPdSuite, 76, "testOralSplit" ) {}
 void runTest() { suite_LSTMPkPdSuite.testOralSplit(); }
} testDescription_LSTMPkPdSuite_testOralSplit;

static class TestDescription_LSTMPkPdSuite_testOralDecayed : public CxxTest::RealTestDescription {
public:
 TestDescription_LSTMPkPdSuite_testOralDecayed() : CxxTest::RealTestDescription( Tests_LSTMPkPdSuite, suiteDescription_LSTMPkPdSuite, 82, "testOralDecayed" ) {}
 void runTest() { suite_LSTMPkPdSuite.testOralDecayed(); }
} testDescription_LSTMPkPdSuite_testOralDecayed;

static class TestDescription_LSTMPkPdSuite_testOral2Doses : public CxxTest::RealTestDescription {
public:
 TestDescription_LSTMPkPdSuite_testOral2Doses() : CxxTest::RealTestDescription( Tests_LSTMPkPdSuite, suiteDescription_LSTMPkPdSuite, 88, "testOral2Doses" ) {}
 void runTest() { suite_LSTMPkPdSuite.testOral2Doses(); }
} testDescription_LSTMPkPdSuite_testOral2Doses;

#include "CheckpointSuite.h"

static CheckpointSuite suite_CheckpointSuite;

static CxxTest::List Tests_CheckpointSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_CheckpointSuite( "CheckpointSuite.h", 34, "CheckpointSuite", suite_CheckpointSuite, Tests_CheckpointSuite );

static class TestDescription_CheckpointSuite_testCheckpointing : public CxxTest::RealTestDescription {
public:
 TestDescription_CheckpointSuite_testCheckpointing() : CxxTest::RealTestDescription( Tests_CheckpointSuite, suiteDescription_CheckpointSuite, 47, "testCheckpointing" ) {}
 void runTest() { suite_CheckpointSuite.testCheckpointing(); }
} testDescription_CheckpointSuite_testCheckpointing;

#include "DummyInfectionSuite.h"

static DummyInfectionSuite suite_DummyInfectionSuite;

static CxxTest::List Tests_DummyInfectionSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_DummyInfectionSuite( "DummyInfectionSuite.h", 34, "DummyInfectionSuite", suite_DummyInfectionSuite, Tests_DummyInfectionSuite );

static class TestDescription_DummyInfectionSuite_testNewInf : public CxxTest::RealTestDescription {
public:
 TestDescription_DummyInfectionSuite_testNewInf() : CxxTest::RealTestDescription( Tests_DummyInfectionSuite, suiteDescription_DummyInfectionSuite, 53, "testNewInf" ) {}
 void runTest() { suite_DummyInfectionSuite.testNewInf(); }
} testDescription_DummyInfectionSuite_testNewInf;

static class TestDescription_DummyInfectionSuite_testUpdatedInf : public CxxTest::RealTestDescription {
public:
 TestDescription_DummyInfectionSuite_testUpdatedInf() : CxxTest::RealTestDescription( Tests_DummyInfectionSuite, suiteDescription_DummyInfectionSuite, 57, "testUpdatedInf" ) {}
 void runTest() { suite_DummyInfectionSuite.testUpdatedInf(); }
} testDescription_DummyInfectionSuite_testUpdatedInf;

static class TestDescription_DummyInfectionSuite_testUpdated2Inf : public CxxTest::RealTestDescription {
public:
 TestDescription_DummyInfectionSuite_testUpdated2Inf() : CxxTest::RealTestDescription( Tests_DummyInfectionSuite, suiteDescription_DummyInfectionSuite, 62, "testUpdated2Inf" ) {}
 void runTest() { suite_DummyInfectionSuite.testUpdated2Inf(); }
} testDescription_DummyInfectionSuite_testUpdated2Inf;

static class TestDescription_DummyInfectionSuite_testUpdatedReducedInf : public CxxTest::RealTestDescription {
public:
 TestDescription_DummyInfectionSuite_testUpdatedReducedInf() : CxxTest::RealTestDescription( Tests_DummyInfectionSuite, suiteDescription_DummyInfectionSuite, 70, "testUpdatedReducedInf" ) {}
 void runTest() { suite_DummyInfectionSuite.testUpdatedReducedInf(); }
} testDescription_DummyInfectionSuite_testUpdatedReducedInf;

static class TestDescription_DummyInfectionSuite_testUpdatedReducedInf2 : public CxxTest::RealTestDescription {
public:
 TestDescription_DummyInfectionSuite_testUpdatedReducedInf2() : CxxTest::RealTestDescription( Tests_DummyInfectionSuite, suiteDescription_DummyInfectionSuite, 78, "testUpdatedReducedInf2" ) {}
 void runTest() { suite_DummyInfectionSuite.testUpdatedReducedInf2(); }
} testDescription_DummyInfectionSuite_testUpdatedReducedInf2;

#include "EmpiricalInfectionSuite.h"

static EmpiricalInfectionSuite suite_EmpiricalInfectionSuite;

static CxxTest::List Tests_EmpiricalInfectionSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_EmpiricalInfectionSuite( "EmpiricalInfectionSuite.h", 35, "EmpiricalInfectionSuite", suite_EmpiricalInfectionSuite, Tests_EmpiricalInfectionSuite );

static class TestDescription_EmpiricalInfectionSuite_testNewInf : public CxxTest::RealTestDescription {
public:
 TestDescription_EmpiricalInfectionSuite_testNewInf() : CxxTest::RealTestDescription( Tests_EmpiricalInfectionSuite, suiteDescription_EmpiricalInfectionSuite, 55, "testNewInf" ) {}
 void runTest() { suite_EmpiricalInfectionSuite.testNewInf(); }
} testDescription_EmpiricalInfectionSuite_testNewInf;

static class TestDescription_EmpiricalInfectionSuite_testUpdatedInf : public CxxTest::RealTestDescription {
public:
 TestDescription_EmpiricalInfectionSuite_testUpdatedInf() : CxxTest::RealTestDescription( Tests_EmpiricalInfectionSuite, suiteDescription_EmpiricalInfectionSuite, 60, "testUpdatedInf" ) {}
 void runTest() { suite_EmpiricalInfectionSuite.testUpdatedInf(); }
} testDescription_EmpiricalInfectionSuite_testUpdatedInf;

static class TestDescription_EmpiricalInfectionSuite_testUpdated2Inf : public CxxTest::RealTestDescription {
public:
 TestDescription_EmpiricalInfectionSuite_testUpdated2Inf() : CxxTest::RealTestDescription( Tests_EmpiricalInfectionSuite, suiteDescription_EmpiricalInfectionSuite, 65, "testUpdated2Inf" ) {}
 void runTest() { suite_EmpiricalInfectionSuite.testUpdated2Inf(); }
} testDescription_EmpiricalInfectionSuite_testUpdated2Inf;

static class TestDescription_EmpiricalInfectionSuite_testUpdated3Inf : public CxxTest::RealTestDescription {
public:
 TestDescription_EmpiricalInfectionSuite_testUpdated3Inf() : CxxTest::RealTestDescription( Tests_EmpiricalInfectionSuite, suiteDescription_EmpiricalInfectionSuite, 72, "testUpdated3Inf" ) {}
 void runTest() { suite_EmpiricalInfectionSuite.testUpdated3Inf(); }
} testDescription_EmpiricalInfectionSuite_testUpdated3Inf;

static class TestDescription_EmpiricalInfectionSuite_testUpdated4Inf : public CxxTest::RealTestDescription {
public:
 TestDescription_EmpiricalInfectionSuite_testUpdated4Inf() : CxxTest::RealTestDescription( Tests_EmpiricalInfectionSuite, suiteDescription_EmpiricalInfectionSuite, 81, "testUpdated4Inf" ) {}
 void runTest() { suite_EmpiricalInfectionSuite.testUpdated4Inf(); }
} testDescription_EmpiricalInfectionSuite_testUpdated4Inf;

static class TestDescription_EmpiricalInfectionSuite_testUpdatedInf1 : public CxxTest::RealTestDescription {
public:
 TestDescription_EmpiricalInfectionSuite_testUpdatedInf1() : CxxTest::RealTestDescription( Tests_EmpiricalInfectionSuite, suiteDescription_EmpiricalInfectionSuite, 92, "testUpdatedInf1" ) {}
 void runTest() { suite_EmpiricalInfectionSuite.testUpdatedInf1(); }
} testDescription_EmpiricalInfectionSuite_testUpdatedInf1;

static class TestDescription_EmpiricalInfectionSuite_testUpdatedReducedInf : public CxxTest::RealTestDescription {
public:
 TestDescription_EmpiricalInfectionSuite_testUpdatedReducedInf() : CxxTest::RealTestDescription( Tests_EmpiricalInfectionSuite, suiteDescription_EmpiricalInfectionSuite, 98, "testUpdatedReducedInf" ) {}
 void runTest() { suite_EmpiricalInfectionSuite.testUpdatedReducedInf(); }
} testDescription_EmpiricalInfectionSuite_testUpdatedReducedInf;

static class TestDescription_EmpiricalInfectionSuite_testUpdatedReducedInf2 : public CxxTest::RealTestDescription {
public:
 TestDescription_EmpiricalInfectionSuite_testUpdatedReducedInf2() : CxxTest::RealTestDescription( Tests_EmpiricalInfectionSuite, suiteDescription_EmpiricalInfectionSuite, 106, "testUpdatedReducedInf2" ) {}
 void runTest() { suite_EmpiricalInfectionSuite.testUpdatedReducedInf2(); }
} testDescription_EmpiricalInfectionSuite_testUpdatedReducedInf2;

#include "InfectionImmunitySuite.h"

static InfectionImmunitySuite suite_InfectionImmunitySuite;

static CxxTest::List Tests_InfectionImmunitySuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_InfectionImmunitySuite( "InfectionImmunitySuite.h", 31, "InfectionImmunitySuite", suite_InfectionImmunitySuite, Tests_InfectionImmunitySuite );

static class TestDescription_InfectionImmunitySuite_testImmunity : public CxxTest::RealTestDescription {
public:
 TestDescription_InfectionImmunitySuite_testImmunity() : CxxTest::RealTestDescription( Tests_InfectionImmunitySuite, suiteDescription_InfectionImmunitySuite, 42, "testImmunity" ) {}
 void runTest() { suite_InfectionImmunitySuite.testImmunity(); }
} testDescription_InfectionImmunitySuite_testImmunity;

#include "CMDecisionTreeSuite.h"

static CMDecisionTreeSuite suite_CMDecisionTreeSuite;

static CxxTest::List Tests_CMDecisionTreeSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_CMDecisionTreeSuite( "CMDecisionTreeSuite.h", 39, "CMDecisionTreeSuite", suite_CMDecisionTreeSuite, Tests_CMDecisionTreeSuite );

static class TestDescription_CMDecisionTreeSuite_testRandomP : public CxxTest::RealTestDescription {
public:
 TestDescription_CMDecisionTreeSuite_testRandomP() : CxxTest::RealTestDescription( Tests_CMDecisionTreeSuite, suiteDescription_CMDecisionTreeSuite, 89, "testRandomP" ) {}
 void runTest() { suite_CMDecisionTreeSuite.testRandomP(); }
} testDescription_CMDecisionTreeSuite_testRandomP;

static class TestDescription_CMDecisionTreeSuite_testUC2Test : public CxxTest::RealTestDescription {
public:
 TestDescription_CMDecisionTreeSuite_testUC2Test() : CxxTest::RealTestDescription( Tests_CMDecisionTreeSuite, suiteDescription_CMDecisionTreeSuite, 126, "testUC2Test" ) {}
 void runTest() { suite_CMDecisionTreeSuite.testUC2Test(); }
} testDescription_CMDecisionTreeSuite_testUC2Test;

static class TestDescription_CMDecisionTreeSuite_testParasiteTest : public CxxTest::RealTestDescription {
public:
 TestDescription_CMDecisionTreeSuite_testParasiteTest() : CxxTest::RealTestDescription( Tests_CMDecisionTreeSuite, suiteDescription_CMDecisionTreeSuite, 145, "testParasiteTest" ) {}
 void runTest() { suite_CMDecisionTreeSuite.testParasiteTest(); }
} testDescription_CMDecisionTreeSuite_testParasiteTest;

static class TestDescription_CMDecisionTreeSuite_testAgeSwitch : public CxxTest::RealTestDescription {
public:
 TestDescription_CMDecisionTreeSuite_testAgeSwitch() : CxxTest::RealTestDescription( Tests_CMDecisionTreeSuite, suiteDescription_CMDecisionTreeSuite, 181, "testAgeSwitch" ) {}
 void runTest() { suite_CMDecisionTreeSuite.testAgeSwitch(); }
} testDescription_CMDecisionTreeSuite_testAgeSwitch;

static class TestDescription_CMDecisionTreeSuite_testSimpleTreat : public CxxTest::RealTestDescription {
public:
 TestDescription_CMDecisionTreeSuite_testSimpleTreat() : CxxTest::RealTestDescription( Tests_CMDecisionTreeSuite, suiteDescription_CMDecisionTreeSuite, 210, "testSimpleTreat" ) {}
 void runTest() { suite_CMDecisionTreeSuite.testSimpleTreat(); }
} testDescription_CMDecisionTreeSuite_testSimpleTreat;

static class TestDescription_CMDecisionTreeSuite_testDosing : public CxxTest::RealTestDescription {
public:
 TestDescription_CMDecisionTreeSuite_testDosing() : CxxTest::RealTestDescription( Tests_CMDecisionTreeSuite, suiteDescription_CMDecisionTreeSuite, 238, "testDosing" ) {}
 void runTest() { suite_CMDecisionTreeSuite.testDosing(); }
} testDescription_CMDecisionTreeSuite_testDosing;

#include "AgeGroupInterpolationSuite.h"

static AgeGroupInterpolationSuite suite_AgeGroupInterpolationSuite;

static CxxTest::List Tests_AgeGroupInterpolationSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_AgeGroupInterpolationSuite( "AgeGroupInterpolationSuite.h", 32, "AgeGroupInterpolationSuite", suite_AgeGroupInterpolationSuite, Tests_AgeGroupInterpolationSuite );

static class TestDescription_AgeGroupInterpolationSuite_testDummy : public CxxTest::RealTestDescription {
public:
 TestDescription_AgeGroupInterpolationSuite_testDummy() : CxxTest::RealTestDescription( Tests_AgeGroupInterpolationSuite, suiteDescription_AgeGroupInterpolationSuite, 54, "testDummy" ) {}
 void runTest() { suite_AgeGroupInterpolationSuite.testDummy(); }
} testDescription_AgeGroupInterpolationSuite_testDummy;

static class TestDescription_AgeGroupInterpolationSuite_testPiecewiseConst : public CxxTest::RealTestDescription {
public:
 TestDescription_AgeGroupInterpolationSuite_testPiecewiseConst() : CxxTest::RealTestDescription( Tests_AgeGroupInterpolationSuite, suiteDescription_AgeGroupInterpolationSuite, 59, "testPiecewiseConst" ) {}
 void runTest() { suite_AgeGroupInterpolationSuite.testPiecewiseConst(); }
} testDescription_AgeGroupInterpolationSuite_testPiecewiseConst;

static class TestDescription_AgeGroupInterpolationSuite_testLinearInterp : public CxxTest::RealTestDescription {
public:
 TestDescription_AgeGroupInterpolationSuite_testLinearInterp() : CxxTest::RealTestDescription( Tests_AgeGroupInterpolationSuite, suiteDescription_AgeGroupInterpolationSuite, 68, "testLinearInterp" ) {}
 void runTest() { suite_AgeGroupInterpolationSuite.testLinearInterp(); }
} testDescription_AgeGroupInterpolationSuite_testLinearInterp;

#include "DecayFunctionSuite.h"

static DecayFunctionSuite suite_DecayFunctionSuite;

static CxxTest::List Tests_DecayFunctionSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_DecayFunctionSuite( "DecayFunctionSuite.h", 33, "DecayFunctionSuite", suite_DecayFunctionSuite, Tests_DecayFunctionSuite );

static class TestDescription_DecayFunctionSuite_testBad : public CxxTest::RealTestDescription {
public:
 TestDescription_DecayFunctionSuite_testBad() : CxxTest::RealTestDescription( Tests_DecayFunctionSuite, suiteDescription_DecayFunctionSuite, 47, "testBad" ) {}
 void runTest() { suite_DecayFunctionSuite.testBad(); }
} testDescription_DecayFunctionSuite_testBad;

static class TestDescription_DecayFunctionSuite_testConstant : public CxxTest::RealTestDescription {
public:
 TestDescription_DecayFunctionSuite_testConstant() : CxxTest::RealTestDescription( Tests_DecayFunctionSuite, suiteDescription_DecayFunctionSuite, 54, "testConstant" ) {}
 void runTest() { suite_DecayFunctionSuite.testConstant(); }
} testDescription_DecayFunctionSuite_testConstant;

static class TestDescription_DecayFunctionSuite_testStep : public CxxTest::RealTestDescription {
public:
 TestDescription_DecayFunctionSuite_testStep() : CxxTest::RealTestDescription( Tests_DecayFunctionSuite, suiteDescription_DecayFunctionSuite, 68, "testStep" ) {}
 void runTest() { suite_DecayFunctionSuite.testStep(); }
} testDescription_DecayFunctionSuite_testStep;

static class TestDescription_DecayFunctionSuite_testLinear : public CxxTest::RealTestDescription {
public:
 TestDescription_DecayFunctionSuite_testLinear() : CxxTest::RealTestDescription( Tests_DecayFunctionSuite, suiteDescription_DecayFunctionSuite, 80, "testLinear" ) {}
 void runTest() { suite_DecayFunctionSuite.testLinear(); }
} testDescription_DecayFunctionSuite_testLinear;

static class TestDescription_DecayFunctionSuite_testExponential : public CxxTest::RealTestDescription {
public:
 TestDescription_DecayFunctionSuite_testExponential() : CxxTest::RealTestDescription( Tests_DecayFunctionSuite, suiteDescription_DecayFunctionSuite, 91, "testExponential" ) {}
 void runTest() { suite_DecayFunctionSuite.testExponential(); }
} testDescription_DecayFunctionSuite_testExponential;

static class TestDescription_DecayFunctionSuite_testWeibull : public CxxTest::RealTestDescription {
public:
 TestDescription_DecayFunctionSuite_testWeibull() : CxxTest::RealTestDescription( Tests_DecayFunctionSuite, suiteDescription_DecayFunctionSuite, 102, "testWeibull" ) {}
 void runTest() { suite_DecayFunctionSuite.testWeibull(); }
} testDescription_DecayFunctionSuite_testWeibull;

static class TestDescription_DecayFunctionSuite_testHill : public CxxTest::RealTestDescription {
public:
 TestDescription_DecayFunctionSuite_testHill() : CxxTest::RealTestDescription( Tests_DecayFunctionSuite, suiteDescription_DecayFunctionSuite, 111, "testHill" ) {}
 void runTest() { suite_DecayFunctionSuite.testHill(); }
} testDescription_DecayFunctionSuite_testHill;

static class TestDescription_DecayFunctionSuite_testSmoothCompact : public CxxTest::RealTestDescription {
public:
 TestDescription_DecayFunctionSuite_testSmoothCompact() : CxxTest::RealTestDescription( Tests_DecayFunctionSuite, suiteDescription_DecayFunctionSuite, 122, "testSmoothCompact" ) {}
 void runTest() { suite_DecayFunctionSuite.testSmoothCompact(); }
} testDescription_DecayFunctionSuite_testSmoothCompact;

#include "PennyInfectionSuite.h"

static PennyInfectionSuite suite_PennyInfectionSuite;

static CxxTest::List Tests_PennyInfectionSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_PennyInfectionSuite( "PennyInfectionSuite.h", 36, "PennyInfectionSuite", suite_PennyInfectionSuite, Tests_PennyInfectionSuite );

static class TestDescription_PennyInfectionSuite_testThresholds : public CxxTest::RealTestDescription {
public:
 TestDescription_PennyInfectionSuite_testThresholds() : CxxTest::RealTestDescription( Tests_PennyInfectionSuite, suiteDescription_PennyInfectionSuite, 51, "testThresholds" ) {}
 void runTest() { suite_PennyInfectionSuite.testThresholds(); }
} testDescription_PennyInfectionSuite_testThresholds;

static class TestDescription_PennyInfectionSuite_testDensities : public CxxTest::RealTestDescription {
public:
 TestDescription_PennyInfectionSuite_testDensities() : CxxTest::RealTestDescription( Tests_PennyInfectionSuite, suiteDescription_PennyInfectionSuite, 67, "testDensities" ) {}
 void runTest() { suite_PennyInfectionSuite.testDensities(); }
} testDescription_PennyInfectionSuite_testDensities;

#include "MolineauxInfectionSuite.h"

static MolineauxInfectionSuite suite_MolineauxInfectionSuite;

static CxxTest::List Tests_MolineauxInfectionSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_MolineauxInfectionSuite( "MolineauxInfectionSuite.h", 38, "MolineauxInfectionSuite", suite_MolineauxInfectionSuite, Tests_MolineauxInfectionSuite );

static class TestDescription_MolineauxInfectionSuite_testDensities : public CxxTest::RealTestDescription {
public:
 TestDescription_MolineauxInfectionSuite_testDensities() : CxxTest::RealTestDescription( Tests_MolineauxInfectionSuite, suiteDescription_MolineauxInfectionSuite, 61, "testDensities" ) {}
 void runTest() { suite_MolineauxInfectionSuite.testDensities(); }
} testDescription_MolineauxInfectionSuite_testDensities;

static class TestDescription_MolineauxInfectionSuite_testMolOrig : public CxxTest::RealTestDescription {
public:
 TestDescription_MolineauxInfectionSuite_testMolOrig() : CxxTest::RealTestDescription( Tests_MolineauxInfectionSuite, suiteDescription_MolineauxInfectionSuite, 91, "testMolOrig" ) {}
 void runTest() { suite_MolineauxInfectionSuite.testMolOrig(); }
} testDescription_MolineauxInfectionSuite_testMolOrig;

static class TestDescription_MolineauxInfectionSuite_testMolOrigRG : public CxxTest::RealTestDescription {
public:
 TestDescription_MolineauxInfectionSuite_testMolOrigRG() : CxxTest::RealTestDescription( Tests_MolineauxInfectionSuite, suiteDescription_MolineauxInfectionSuite, 101, "testMolOrigRG" ) {}
 void runTest() { suite_MolineauxInfectionSuite.testMolOrigRG(); }
} testDescription_MolineauxInfectionSuite_testMolOrigRG;

static class TestDescription_MolineauxInfectionSuite_testMolPairwise : public CxxTest::RealTestDescription {
public:
 TestDescription_MolineauxInfectionSuite_testMolPairwise() : CxxTest::RealTestDescription( Tests_MolineauxInfectionSuite, suiteDescription_MolineauxInfectionSuite, 165, "testMolPairwise" ) {}
 void runTest() { suite_MolineauxInfectionSuite.testMolPairwise(); }
} testDescription_MolineauxInfectionSuite_testMolPairwise;

static class TestDescription_MolineauxInfectionSuite_testMolPairwiseRG : public CxxTest::RealTestDescription {
public:
 TestDescription_MolineauxInfectionSuite_testMolPairwiseRG() : CxxTest::RealTestDescription( Tests_MolineauxInfectionSuite, suiteDescription_MolineauxInfectionSuite, 176, "testMolPairwiseRG" ) {}
 void runTest() { suite_MolineauxInfectionSuite.testMolPairwiseRG(); }
} testDescription_MolineauxInfectionSuite_testMolPairwiseRG;

#include "UtilVectorsSuite.h"

static UtilVectorsSuite suite_UtilVectorsSuite;

static CxxTest::List Tests_UtilVectorsSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_UtilVectorsSuite( "UtilVectorsSuite.h", 34, "UtilVectorsSuite", suite_UtilVectorsSuite, Tests_UtilVectorsSuite );

static class TestDescription_UtilVectorsSuite_testInversion : public CxxTest::RealTestDescription {
public:
 TestDescription_UtilVectorsSuite_testInversion() : CxxTest::RealTestDescription( Tests_UtilVectorsSuite, suiteDescription_UtilVectorsSuite, 37, "testInversion" ) {}
 void runTest() { suite_UtilVectorsSuite.testInversion(); }
} testDescription_UtilVectorsSuite_testInversion;

#include "PkPdComplianceSuite.h"

static PkPdComplianceSuite suite_PkPdComplianceSuite;

static CxxTest::List Tests_PkPdComplianceSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_PkPdComplianceSuite( "PkPdComplianceSuite.h", 56, "PkPdComplianceSuite", suite_PkPdComplianceSuite, Tests_PkPdComplianceSuite );

static class TestDescription_PkPdComplianceSuite_testAR1 : public CxxTest::RealTestDescription {
public:
 TestDescription_PkPdComplianceSuite_testAR1() : CxxTest::RealTestDescription( Tests_PkPdComplianceSuite, suiteDescription_PkPdComplianceSuite, 240, "testAR1" ) {}
 void runTest() { suite_PkPdComplianceSuite.testAR1(); }
} testDescription_PkPdComplianceSuite_testAR1;

static class TestDescription_PkPdComplianceSuite_testAR : public CxxTest::RealTestDescription {
public:
 TestDescription_PkPdComplianceSuite_testAR() : CxxTest::RealTestDescription( Tests_PkPdComplianceSuite, suiteDescription_PkPdComplianceSuite, 248, "testAR" ) {}
 void runTest() { suite_PkPdComplianceSuite.testAR(); }
} testDescription_PkPdComplianceSuite_testAR;

static class TestDescription_PkPdComplianceSuite_testAS1 : public CxxTest::RealTestDescription {
public:
 TestDescription_PkPdComplianceSuite_testAS1() : CxxTest::RealTestDescription( Tests_PkPdComplianceSuite, suiteDescription_PkPdComplianceSuite, 257, "testAS1" ) {}
 void runTest() { suite_PkPdComplianceSuite.testAS1(); }
} testDescription_PkPdComplianceSuite_testAS1;

static class TestDescription_PkPdComplianceSuite_testAS : public CxxTest::RealTestDescription {
public:
 TestDescription_PkPdComplianceSuite_testAS() : CxxTest::RealTestDescription( Tests_PkPdComplianceSuite, suiteDescription_PkPdComplianceSuite, 265, "testAS" ) {}
 void runTest() { suite_PkPdComplianceSuite.testAS(); }
} testDescription_PkPdComplianceSuite_testAS;

static class TestDescription_PkPdComplianceSuite_testCQ : public CxxTest::RealTestDescription {
public:
 TestDescription_PkPdComplianceSuite_testCQ() : CxxTest::RealTestDescription( Tests_PkPdComplianceSuite, suiteDescription_PkPdComplianceSuite, 275, "testCQ" ) {}
 void runTest() { suite_PkPdComplianceSuite.testCQ(); }
} testDescription_PkPdComplianceSuite_testCQ;

static class TestDescription_PkPdComplianceSuite_testDHA : public CxxTest::RealTestDescription {
public:
 TestDescription_PkPdComplianceSuite_testDHA() : CxxTest::RealTestDescription( Tests_PkPdComplianceSuite, suiteDescription_PkPdComplianceSuite, 283, "testDHA" ) {}
 void runTest() { suite_PkPdComplianceSuite.testDHA(); }
} testDescription_PkPdComplianceSuite_testDHA;

static class TestDescription_PkPdComplianceSuite_testLF : public CxxTest::RealTestDescription {
public:
 TestDescription_PkPdComplianceSuite_testLF() : CxxTest::RealTestDescription( Tests_PkPdComplianceSuite, suiteDescription_PkPdComplianceSuite, 291, "testLF" ) {}
 void runTest() { suite_PkPdComplianceSuite.testLF(); }
} testDescription_PkPdComplianceSuite_testLF;

static class TestDescription_PkPdComplianceSuite_testMQ : public CxxTest::RealTestDescription {
public:
 TestDescription_PkPdComplianceSuite_testMQ() : CxxTest::RealTestDescription( Tests_PkPdComplianceSuite, suiteDescription_PkPdComplianceSuite, 299, "testMQ" ) {}
 void runTest() { suite_PkPdComplianceSuite.testMQ(); }
} testDescription_PkPdComplianceSuite_testMQ;

static class TestDescription_PkPdComplianceSuite_testPPQ1C : public CxxTest::RealTestDescription {
public:
 TestDescription_PkPdComplianceSuite_testPPQ1C() : CxxTest::RealTestDescription( Tests_PkPdComplianceSuite, suiteDescription_PkPdComplianceSuite, 308, "testPPQ1C" ) {}
 void runTest() { suite_PkPdComplianceSuite.testPPQ1C(); }
} testDescription_PkPdComplianceSuite_testPPQ1C;

static class TestDescription_PkPdComplianceSuite_testPPQ_Hodel2013 : public CxxTest::RealTestDescription {
public:
 TestDescription_PkPdComplianceSuite_testPPQ_Hodel2013() : CxxTest::RealTestDescription( Tests_PkPdComplianceSuite, suiteDescription_PkPdComplianceSuite, 317, "testPPQ_Hodel2013" ) {}
 void runTest() { suite_PkPdComplianceSuite.testPPQ_Hodel2013(); }
} testDescription_PkPdComplianceSuite_testPPQ_Hodel2013;

static class TestDescription_PkPdComplianceSuite_testPPQ_Tarning2012AAC : public CxxTest::RealTestDescription {
public:
 TestDescription_PkPdComplianceSuite_testPPQ_Tarning2012AAC() : CxxTest::RealTestDescription( Tests_PkPdComplianceSuite, suiteDescription_PkPdComplianceSuite, 326, "testPPQ_Tarning2012AAC" ) {}
 void runTest() { suite_PkPdComplianceSuite.testPPQ_Tarning2012AAC(); }
} testDescription_PkPdComplianceSuite_testPPQ_Tarning2012AAC;

#include <cxxtest/Root.cpp>
