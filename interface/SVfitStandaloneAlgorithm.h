#ifndef TauAnalysis_SVfitStandalone_SVfitStandaloneAlgorithm_h
#define TauAnalysis_SVfitStandalone_SVfitStandaloneAlgorithm_h

#include "TauAnalysis/SVfitStandalone/interface/SVfitStandaloneLikelihood.h"
#include "TauAnalysis/SVfitStandalone/interface/SVfitStandaloneMarkovChainIntegrator.h"
#include "TauAnalysis/SVfitStandalone/interface/svFitStandaloneAuxFunctions.h"
#include "TauAnalysis/SVfitStandalone/interface/SVfitStandaloneQuantities.h"

#include <TMath.h>
#include <TArrayF.h>
#include <TString.h>
#include <TH1.h>
#include <TBenchmark.h>

using svFitStandalone::Vector;
using svFitStandalone::LorentzVector;
using svFitStandalone::MeasuredTauLepton;


/**
   \class   SVfitStandaloneAlgorithm SVfitStandaloneAlgorithm.h "TauAnalysis/SVfitStandalone/interface/SVfitStandaloneAlgorithm.h"
   
   \brief   Standalone version of the SVfitAlgorithm.

   This class is a standalone version of the SVfitAlgorithm to perform the full reconstruction of a di-tau resonance system. The 
   implementation is supposed to deal with any combination of leptonic or hadronic tau decays. It exploits likelihood functions 
   as defined in interface/LikelihoodFunctions.h of this package, which are combined into a single likelihood function as defined 
   interface/SVfitStandaloneLikelihood.h in this package. The combined likelihood function depends on the following variables: 

   \var nunuMass   : the invariant mass of the neutrino system for each decay branch (two parameters)
   \var decayAngle : the decay angle in the restframe of each decay branch (two parameters)
   \var visMass    : the mass of the visible component of the di-tau system (two parameters)

   The actual fit parameters are:

   \var nunuMass   : the invariant mass of the neutrino system for each decay branch (two parameters)
   \var xFrac      : the fraction of the visible energy on the energy of the tau lepton in the labframe (two parameters)
   \var phi        : the azimuthal angle of the tau lepton (two parameters)

   In the fit mode. The azimuthal angle of each tau lepton is not constraint by measurement. It is limited to the physical values 
   from -Math::Phi to Math::Phi in the likelihood function of the combined likelihood class. The parameter nunuMass is constraint 
   to the tau lepton mass minus the mass of the visible part of the decay (which is itself constraint to values below the tau 
   lepton mass) in the setup function of this class. The parameter xFrac is constraint to values between 0. and 1. in the setup 
   function of this class. The invariant mass of the neutrino system is fixed to be zero for hadronic tau lepton decays as only 
   one (tau-) neutrino is involved in the decay. The original number of free parameters of 6 is therefore reduced by one for each 
   hadronic tau decay within the resonance. All information about the negative log likelihood is stored in the SVfitStandaloneLikelihood 
   class as defined in the same package. This class interfaces the combined likelihood to the ROOT::Math::Minuit minimization program. 
   It does setup/initialize the fit parameters as defined in interface/SVfitStandaloneLikelihood.h in this package, initializes the 
   minimization procedure, executes the fit algorithm and returns the fit result. The fit result consists of the fully reconstructed 
   di-tau system, from which also the invariant mass can be derived.

   In the integration mode xFrac for the second leptons is determiend from xFrac of the first lepton for given di-tau mass, thus reducing 
   the number of parameters to be integrated out wrt. to the fit version by one. The di-tau mass is scanned for the highest likelihood 
   starting from the visible mass of the two leptons. The return value is just the di-tau mass. 

   Common usage is: 
   
   // construct the class object from the minimal necessary information
   SVfitStandaloneAlgorithm algo(measuredTauLeptons, measuredMET, covMET);
   // apply customized configurations if wanted (examples are given below)
   //algo.maxObjFunctionCalls(10000); // only applies for fit mode
   //algo.addLogM(false);             // applies for fit and integration mode
   //algo.metPower(0.5);              // only applies for fit mode
   // run the fit in fit mode
   algo.fit();
   // retrieve the results upon success
   if ( algo.isValidSolution() ) {
     std::cout << algo.mass();
   }
   // run the integration in integration mode
   algo.integrate();
   std::cout << algo.mass();

   The following optional parameters can be applied after initialization but before running the fit in fit mode: 

   \var metPower : indicating an additional power to enhance the MET likelihood (default is 1.)
   \var addLogM : specifying whether to use the LogM penalty term or not (default is true)     
   \var maxObjFunctionCalls : the maximum of function calls before the minimization procedure is terminated (default is 5000)
*/

class SVfitStandaloneAlgorithm
{
 public:
  /// constructor from a minimal set of configurables
  SVfitStandaloneAlgorithm(const std::vector<MeasuredTauLepton>& measuredTauLeptons, double measuredMETx, double measuredMETy, const TMatrixD& covMET, unsigned int verbosity = 0);
  /// destructor
  ~SVfitStandaloneAlgorithm();

  /// add an additional logM(tau,tau) term to the nll to suppress tails on M(tau,tau) (default is false)
  void addLogM(bool value, double power = 1.) { nll_->addLogM(value, power); }
  /// modify the MET term in the nll by an additional power (default is 1.)
  void metPower(double value) { nll_->metPower(value); }
  /// marginalize unknown mass of hadronic tau decay products (ATLAS case)
  void marginalizeVisMass(bool value, TFile* inputFile);
  void marginalizeVisMass(bool value, const TH1*);    
  /// take resolution on energy and mass of hadronic tau decays into account
  void shiftVisMass(bool value, TFile* inputFile);
  void shiftVisPt(bool value, TFile* inputFile);
  /// maximum function calls after which to stop the minimization procedure (default is 5000)
  void maxObjFunctionCalls(double value) { maxObjFunctionCalls_ = value; }

  /// fit to be called from outside
  void fit();
  /// integration by VEGAS (kept for legacy)
  void integrate() { return integrateVEGAS(""); }
  /// integration by VEGAS to be called from outside
  void integrateVEGAS(const std::string& likelihoodFileName = "");
  /// integration by Markov Chain MC to be called from outside
  void integrateMarkovChain(const std::string& likelihoodFileName = "");

  /// return status of minuit fit
  /*    
      0: Valid solution
      1: Covariance matrix was made positive definite
      2: Hesse matrix is invalid
      3: Estimated distance to minimum (EDM) is above maximum
      4: Reached maximum number of function calls before reaching convergence
      5: Any other failure
  */
  int fitStatus() { return fitStatus_; }
  /// return whether this is a valid solution or not
  bool isValidSolution() { return (nllStatus_ == 0 && fitStatus_ <= 0); }
  /// return whether this is a valid solution or not
  bool isValidFit() { return fitStatus_ == 0; }
  /// return whether this is a valid solution or not
  bool isValidNLL() { return nllStatus_ == 0; }
  /// return mass of the di-tau system 
  double mass() const { return mass_; }
  /// return uncertainty on the mass of the fitted di-tau system
  double massUncert() const { return massUncert_; }
  /// return transverse mass of the di-tau system 
  double transverseMass() const { return transverseMass_; }
  /// return uncertainty on the transverse mass of the fitted di-tau system
  double transverseMassUncert() const { return transverseMassUncert_; }
  /// return maxima of likelihood scan
  double massLmax() const { return massLmax_; }
  double transverseMassLmax() const { return transverseMassLmax_; }

  /// return mass of the di-tau system (kept for legacy)
  double getMass() const {return mass(); }
  
  void setMCQuantitiesAdapter(svFitStandalone::MCQuantitiesAdapter* mvQuantitiesAdapter);
  svFitStandalone::MCQuantitiesAdapter* getMCQuantitiesAdapter() const;

  /// return 4-vectors of measured tau leptons
  std::vector<LorentzVector> measuredTauLeptons() const;
  /// return 4-vector of the measured di-tau system
  LorentzVector measuredDiTauSystem() const { return measuredTauLeptons()[0] + measuredTauLeptons()[1]; }
  // return spacial vector of the measured MET
  Vector measuredMET() const { return nll_->measuredMET(); }

 protected:
  /// setup the starting values for the minimization (default values for the fit parameters are taken from src/SVFitParameters.cc in the same package)
  void setup();

 protected:
  /// return whether this is a valid solution or not
  int fitStatus_;
  /// return whether this is a valid solution or not
  unsigned int nllStatus_;
  /// verbosity level
  unsigned int verbosity_;
  /// stop minimization after a maximal number of function calls
  unsigned int maxObjFunctionCalls_;

  /// minuit instance 
  ROOT::Math::Minimizer* minimizer_;
  /// standalone combined likelihood
  svFitStandalone::SVfitStandaloneLikelihood* nll_;
  /// needed to make the fit function callable from within minuit
  svFitStandalone::ObjectiveFunctionAdapterMINUIT standaloneObjectiveFunctionAdapterMINUIT_;

  /// needed for VEGAS integration
  svFitStandalone::ObjectiveFunctionAdapterVEGAS* standaloneObjectiveFunctionAdapterVEGAS_;   
  
  /// fitted di-tau mass
  double mass_;
  /// uncertainty on the fitted di-tau mass
  double massUncert_;
  /// maxima of di-tau mass likelihood scan
  double massLmax_;
  /// fitted transverse mass
  double transverseMass_;
  /// uncertainty on the fitted transverse mass
  double transverseMassUncert_;
  /// maxima of transverse mass likelihood scan
  double transverseMassLmax_;
  /// fit result for each of the decay branches
  std::vector<svFitStandalone::LorentzVector> fittedTauLeptons_;
  /// fitted di-tau system
  svFitStandalone::LorentzVector fittedDiTauSystem_;

  /// needed for markov chain integration
  svFitStandalone::MCObjectiveFunctionAdapter* mcObjectiveFunctionAdapter_ = nullptr;
  svFitStandalone::MCQuantitiesAdapter* mcQuantitiesAdapter_ = nullptr;
  SVfitStandaloneMarkovChainIntegrator* integrator2_ = nullptr;
  int integrator2_nDim_;
  bool isInitialized2_;
  unsigned maxObjFunctionCalls2_;

  TBenchmark* clock_;

  /// resolution on Pt and mass of hadronic taus
  bool marginalizeVisMass_;
  const TH1* lutVisMassAllDMs_;
  bool shiftVisMass_;
  const TH1* lutVisMassResDM0_;
  const TH1* lutVisMassResDM1_;
  const TH1* lutVisMassResDM10_;
  bool shiftVisPt_;  
  const TH1* lutVisPtResDM0_;
  const TH1* lutVisPtResDM1_;
  const TH1* lutVisPtResDM10_;

  bool l1isLep_;
  int idxFitParLeg1_;
  bool l2isLep_;
  int idxFitParLeg2_;
};

inline
std::vector<svFitStandalone::LorentzVector> 
SVfitStandaloneAlgorithm::measuredTauLeptons() const 
{ 
  std::vector<svFitStandalone::LorentzVector> measuredTauLeptons;
  measuredTauLeptons.push_back(nll_->measuredTauLeptons()[0].p4());
  measuredTauLeptons.push_back(nll_->measuredTauLeptons()[1].p4());
  return measuredTauLeptons; 
}

#endif
