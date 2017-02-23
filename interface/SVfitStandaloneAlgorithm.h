#ifndef TauAnalysis_SVfitStandalone_SVfitStandaloneAlgorithm_h
#define TauAnalysis_SVfitStandalone_SVfitStandaloneAlgorithm_h

#include "TauAnalysis/SVfitStandalone/interface/SVfitStandaloneLikelihood.h"
#include "TauAnalysis/SVfitStandalone/interface/SVfitStandaloneMarkovChainIntegrator.h"
#include "TauAnalysis/SVfitStandalone/interface/svFitStandaloneAuxFunctions.h"

#include <TMath.h>
#include <TArrayF.h>
#include <TString.h>
#include <TH1.h>
#include <TBenchmark.h>

using svFitStandalone::Vector;
using svFitStandalone::LorentzVector;
using svFitStandalone::MeasuredTauLepton;

/**
   \class   ObjectFunctionAdapter SVfitStandaloneAlgorithm.h "TauAnalysis/SVfitStandalone/interface/SVfitStandaloneAlgorithm.h"
   
   \brief   Function interface to minuit.
   
   This class is an interface, which is used as global function pointer of the combined likelihood as defined in src/SVfitStandaloneLikelihood.cc
   to VEGAS or minuit. It is a member of the of the SVfitStandaloneAlgorithm class defined below and is used in SVfitStandalone::fit(), or 
   SVfitStandalone::integrate(), where it is passed on to a ROOT::Math::Functor. The parameters x correspond to the array of fit/integration 
   paramters as defined in interface/SVfitStandaloneLikelihood.h of this package. In the fit mode these are made known to minuit in the function
   SVfitStandaloneAlgorithm::setup. In the integration mode the mapping is done internally in the SVfitStandaloneLikelihood::tansformint. This
   has to be in sync. with the definition of the integration boundaries in SVfitStandaloneAlgorithm::integrate. 
*/

namespace svFitStandalone
{
  TH1* makeHistogram(const std::string&, double, double, double);
  void compHistogramDensity(const TH1*, TH1*);
  double extractValue(const TH1*, TH1*);
  double extractUncertainty(const TH1*, TH1*);
  double extractLmax(const TH1*, TH1*);

  // for "fit" (MINUIT) mode
  class ObjectiveFunctionAdapterMINUIT
  {
  public:
    double operator()(const double* x) const // NOTE: return value = -log(likelihood)
    {
      double prob = SVfitStandaloneLikelihood::gSVfitStandaloneLikelihood->prob(x);
      double nll;
      if ( prob > 0. ) nll = -TMath::Log(prob);
      else nll = std::numeric_limits<float>::max();
      return nll;
    }
  };
  // for VEGAS integration
  void map_xVEGAS(const double*, bool, bool, bool, bool, bool, double, double, double*);
  class ObjectiveFunctionAdapterVEGAS
  {
  public:
    double Eval(const double* x) const // NOTE: return value = likelihood, **not** -log(likelihood)
    {
      map_xVEGAS(x, l1isLep_, l2isLep_, marginalizeVisMass_, shiftVisMass_, shiftVisPt_, mvis_, mtest_, x_mapped_);      
      double prob = SVfitStandaloneLikelihood::gSVfitStandaloneLikelihood->prob(x_mapped_, true, mtest_);
      if ( TMath::IsNaN(prob) ) prob = 0.;
      return prob;
    }
    void SetL1isLep(bool l1isLep) { l1isLep_ = l1isLep; }
    void SetL2isLep(bool l2isLep) { l2isLep_ = l2isLep; }
    void SetMarginalizeVisMass(bool marginalizeVisMass) { marginalizeVisMass_ = marginalizeVisMass; }
    void SetShiftVisMass(bool shiftVisMass) { shiftVisMass_ = shiftVisMass; }
    void SetShiftVisPt(bool shiftVisPt) { shiftVisPt_ = shiftVisPt; }
    void SetMvis(double mvis) { mvis_ = mvis; }
    void SetMtest(double mtest) { mtest_ = mtest; }
  private:
    mutable double x_mapped_[10];
    bool l1isLep_;
    bool l2isLep_;
    bool marginalizeVisMass_;
    bool shiftVisMass_;
    bool shiftVisPt_;
    double mvis_;  // mass of visible tau decay products
    double mtest_; // current mass hypothesis
  };
  // for Markov Chain integration
  void map_xMarkovChain(const double*, bool, bool, bool, bool, bool, double*);
  class MCObjectiveFunctionAdapter : public ROOT::Math::Functor
  {
   public:
    void SetL1isLep(bool l1isLep) { l1isLep_ = l1isLep; }
    void SetL2isLep(bool l2isLep) { l2isLep_ = l2isLep; }
    void SetMarginalizeVisMass(bool marginalizeVisMass) { marginalizeVisMass_ = marginalizeVisMass; }
    void SetShiftVisMass(bool shiftVisMass) { shiftVisMass_ = shiftVisMass; }
    void SetShiftVisPt(bool shiftVisPt) { shiftVisPt_ = shiftVisPt; }
    void SetNDim(int nDim) { nDim_ = nDim; }
    unsigned int NDim() const { return nDim_; }
   private:
    virtual double DoEval(const double* x) const
    {
      map_xMarkovChain(x, l1isLep_, l2isLep_, marginalizeVisMass_, shiftVisMass_, shiftVisPt_, x_mapped_);
      double prob = SVfitStandaloneLikelihood::gSVfitStandaloneLikelihood->prob(x_mapped_);
      if ( TMath::IsNaN(prob) ) prob = 0.;
      return prob;
    } 
    mutable double x_mapped_[10];
    int nDim_;
    bool l1isLep_;
    bool l2isLep_;
    bool marginalizeVisMass_;
    bool shiftVisMass_;
    bool shiftVisPt_;
  };
  
  class SVfitQuantity
  {
   public:
    SVfitQuantity(TH1* histogram, TH1* histogram_density, std::function<double(std::vector<svFitStandalone::LorentzVector> const&) > function);
    ~SVfitQuantity();
    
    void SetHistograms(TH1* histogram, TH1* histogram_density);
    void Reset();
    void WriteHistograms() const;
    
    double Eval(std::vector<svFitStandalone::LorentzVector> const& fittedTauLeptons) const;
    
    double ExtractValue() const;
    double ExtractUncertainty() const;
    double ExtractLmax() const;
    
    mutable TH1* histogram_ = nullptr;
   
   protected:
    TH1* histogram_density_ = nullptr;
    std::function<double(std::vector<svFitStandalone::LorentzVector> const&) > function_;
  };
  
  class MCQuantitiesAdapter : public ROOT::Math::Functor
  {
   public:
    MCQuantitiesAdapter(std::vector<SVfitQuantity*> const& quantities);
    ~MCQuantitiesAdapter();
    
    void SetHistograms(size_t index, TH1* histogram, TH1* histogram_density);
    void SetHistograms(std::vector<TH1*> histograms, std::vector<TH1*> histogram_densities);
    void SetHistogramMass(TH1* histogram, TH1* histogram_density);
    void SetHistogramTransverseMass(TH1* histogram, TH1* histogram_density);
    void Reset();
    void WriteHistograms() const;
    
    inline void SetL1isLep(bool l1isLep) { l1isLep_ = l1isLep; }
    inline void SetL2isLep(bool l2isLep) { l2isLep_ = l2isLep; }
    inline void SetMarginalizeVisMass(bool marginalizeVisMass) { marginalizeVisMass_ = marginalizeVisMass; }
    inline void SetShiftVisMass(bool shiftVisMass) { shiftVisMass_ = shiftVisMass; }
    inline void SetShiftVisPt(bool shiftVisPt) { shiftVisPt_ = shiftVisPt; }
    void SetNDim(unsigned int nDim) { nDim_ = nDim; }
    
    unsigned int NDim() const { return nDim_; }
    
    double ExtractValue(size_t index) const;
    double ExtractUncertainty(size_t index) const;
    double ExtractLmax(size_t index) const;
    
    std::vector<double> ExtractValues() const;
    std::vector<double> ExtractUncertainties() const;
    std::vector<double> ExtractLmaxima() const;
    
    double getMass() const;
    double getMassUncert() const;
    double getMassLmax() const;
    double getTransverseMass() const;
    double getTransverseMassUncert() const;
    double getTransverseMassLmax() const;
    
   protected:
    std::vector<SVfitQuantity*> quantities_;
    
    mutable std::vector<svFitStandalone::LorentzVector> fittedTauLeptons_;
    mutable double x_mapped_[10];
    bool l1isLep_;
    bool l2isLep_;
    bool marginalizeVisMass_;
    bool shiftVisMass_;
    bool shiftVisPt_;
    unsigned int nDim_;
    
   private:
    virtual double DoEval(const double* x) const;
  };
  
  class MCPtEtaPhiMassAdapter : public MCQuantitiesAdapter
  {
   public:
    MCPtEtaPhiMassAdapter();
    
    double getPt() const;
    double getPtUncert() const;
    double getPtLmax() const;
    double getEta() const;
    double getEtaUncert() const;
    double getEtaLmax() const;
    double getPhi() const;
    double getPhiUncert() const;
    double getPhiLmax() const;
  };
}

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
  
  void setMCQuantitiesAdapter(svFitStandalone::MCQuantitiesAdapter* sVfitMCQuantitiesAdapter);
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
  svFitStandalone::MCQuantitiesAdapter* svfitMCQuantitiesAdapter_ = nullptr;
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
