
/**
   \class testSVfitStandalone testSVfitStandalone.cc "TauAnalysis/SVfitStandalone/bin/testSVfitStandalone.cc"
   \brief Basic example of the use of the standalone version of SVfit

   This is an example executable to show the use of the standalone version of SVfit 
   from a flat n-tuple or single event.
*/

#include "TauAnalysis/SVfitStandalone/interface/SVfitStandaloneAlgorithm.h"

#include "TTree.h"
#include "TFile.h"

void singleEvent()
{
  /* 
     This is a single event for testing in the integration mode.
  */
  // define MET
  Vector MET(11.7491, -51.9172, 0.); 
  // define MET covariance
  TMatrixD covMET(2, 2);
  covMET[0][0] = 787.352;
  covMET[1][0] = -178.63;
  covMET[0][1] = -178.63;
  covMET[1][1] = 179.545;
  // define lepton four vectors
  svFitStandalone::LorentzVector l1( 28.9132, -17.3888, 36.6411, 49.8088); // tau -> electron decay
  svFitStandalone::LorentzVector l2(-24.19  ,  8.77449, 16.9413, 30.8086); // tau -> hadron decay
  std::vector<svFitStandalone::MeasuredTauLepton> measuredTauLeptons;
  measuredTauLeptons.push_back(svFitStandalone::MeasuredTauLepton(svFitStandalone::kTauToElecDecay, l1));
  measuredTauLeptons.push_back(svFitStandalone::MeasuredTauLepton(svFitStandalone::kTauToHadDecay, l2));
  // define algorithm (set the debug level to 3 for testing)
  unsigned verbosity = 2;
  SVfitStandaloneAlgorithm algo(measuredTauLeptons, MET, covMET, verbosity);
  algo.addLogM(false);
  /* 
     the following lines show how to use the different methods on a single event
  */
  // minuit fit method
  //algo.fit();
  // integration by VEGAS (same as function algo.integrate() that has been in use when markov chain integration had not yet been implemented)
  algo.integrateVEGAS();
  // integration by markov chain MC
  //algo.integrateMarkovChain();

  double mass = algo.getMass(); // return value is in units of GeV
  if ( algo.isValidSolution() ) {
    std::cout << "found mass = " << mass << " (expected value = 120.129)" << std::endl;
  } else {
    std::cout << "sorry -- status of NLL is not valid [" << algo.isValidSolution() << "]" << std::endl;
  }
  return;
}

void eventsFromTree(int argc, char* argv[]) 
{
  // parse arguments
  if ( argc < 3 ) {
    std::cout << "Usage : " << argv[0] << " [inputfile.root] [tree_name]" << std::endl;
    return;
  }
  // get intput directory up to one before mass points
  TFile* file = new TFile(argv[1]); 
  // access tree in file
  TTree* tree = (TTree*) file->Get(argv[2]);
  // input variables
  float met, metPhi;
  float covMet11, covMet12; 
  float covMet21, covMet22;
  float l1M, l1Px, l1Py, l1Pz;
  float l2M, l2Px, l2Py, l2Pz;
  float mTrue;
  // branch adresses
  tree->SetBranchAddress("met", &met);
  tree->SetBranchAddress("mphi", &metPhi);
  tree->SetBranchAddress("mcov_11", &covMet11);
  tree->SetBranchAddress("mcov_12", &covMet12);
  tree->SetBranchAddress("mcov_21", &covMet21);
  tree->SetBranchAddress("mcov_22", &covMet22);
  tree->SetBranchAddress("l1_M", &l1M);
  tree->SetBranchAddress("l1_Px", &l1Px);
  tree->SetBranchAddress("l1_Py", &l1Py);
  tree->SetBranchAddress("l1_Pz", &l1Pz);
  tree->SetBranchAddress("l2_M", &l2M);
  tree->SetBranchAddress("l2_Px", &l2Px);
  tree->SetBranchAddress("l2_Py", &l2Py);
  tree->SetBranchAddress("l2_Pz", &l2Pz);
  tree->SetBranchAddress("m_true", &mTrue);
  int nevent = tree->GetEntries();
  for ( int i = 0; i < nevent; ++i ) {
    tree->GetEvent(i);
    std::cout << "event " << (i + 1) << std::endl;
    // setup MET input vector
    svFitStandalone::Vector measuredMET(met *TMath::Sin(metPhi), met *TMath::Cos(metPhi), 0); 
    // setup the MET significance
    TMatrixD covMET(2,2);
    covMET[0][0] = covMet11;
    covMET[0][1] = covMet12;
    covMET[1][0] = covMet21;
    covMET[1][1] = covMet22;
    // setup measure tau lepton vectors 
    svFitStandalone::LorentzVector l1(l1Px, l1Py, l1Pz, TMath::Sqrt(l1M*l1M + l1Px*l1Px + l1Py*l1Py + l1Pz*l1Pz));
    svFitStandalone::LorentzVector l2(l2Px, l2Py, l2Pz, TMath::Sqrt(l2M*l2M + l2Px*l2Px + l2Py*l2Py + l2Pz*l2Pz));
    svFitStandalone::kDecayType l1Type, l2Type;
    if ( std::string(argv[2]) == "EMu" ) {
      l1Type = svFitStandalone::kTauToElecDecay;
      l2Type = svFitStandalone::kTauToMuDecay;
    } else if ( std::string(argv[2]) == "MuTau" ) {
      l1Type = svFitStandalone::kTauToMuDecay;
      l2Type = svFitStandalone::kTauToHadDecay;
    } else if ( std::string(argv[2]) == "ETau" ) {
      l1Type = svFitStandalone::kTauToElecDecay;
      l2Type = svFitStandalone::kTauToHadDecay;
    } else if ( std::string(argv[2]) == "TauTau" ) {
      l1Type = svFitStandalone::kTauToHadDecay;
      l2Type = svFitStandalone::kTauToHadDecay;
    } else {
      std::cerr << "Error: Invalid channel = " << std::string(argv[2]) << " !!" << std::endl;
      std::cerr << "(some customization of this code will be needed for your analysis)" << std::endl;
      assert(0);
    }
    std::vector<svFitStandalone::MeasuredTauLepton> measuredTauLeptons;
    measuredTauLeptons.push_back(svFitStandalone::MeasuredTauLepton(l1Type, l1));
    measuredTauLeptons.push_back(svFitStandalone::MeasuredTauLepton(l2Type, l2));
    // construct the class object from the minimal necesarry information
    SVfitStandaloneAlgorithm algo(measuredTauLeptons, measuredMET, covMET, 1);
    // apply customized configurations if wanted (examples are given below)
    algo.maxObjFunctionCalls(5000);
    //algo.addLogM(false);
    //algo.metPower(0.5)
    // minuit fit method
    //algo.fit();
    // integration by VEGAS (default)
    algo.integrateVEGAS();
    // integration by markov chain MC
    //algo.integrateMarkovChain();
    // retrieve the results upon success
    std::cout << "... m truth : " << mTrue << std::endl;
    if ( algo.isValidSolution() ) {
      std::cout << "... m svfit : " << algo.mass() << " +/- " << algo.massUncert() << std::endl; // return value is in units of GeV
    } else {
      std::cout << "... m svfit : ---" << std::endl;
    }
  }
  return;
}

int main(int argc, char* argv[]) 
{
  //eventsFromTree(argc, argv);
  singleEvent();
  return 0;
}
