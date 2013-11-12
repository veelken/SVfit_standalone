#include "TauAnalysis/SVfitStandalone/interface/SVfitStandaloneAlgorithm.h"

#include "Math/Factory.h"
#include "Math/Functor.h"
#include "Math/GSLMCIntegrator.h"
#include "Math/LorentzVector.h"
#include "Math/PtEtaPhiM4D.h"

#include <TGraphErrors.h>

namespace svFitStandalone
{
  void map_x(const double* x, int nDim, double* x_mapped)
  {
    if(nDim == 4){
      x_mapped[kXFrac]                 = x[0];
      x_mapped[kMNuNu]                 = 0.;
      x_mapped[kPhi]                   = x[1];
      x_mapped[kMaxFitParams + kXFrac] = x[2];
      x_mapped[kMaxFitParams + kMNuNu] = 0.;
      x_mapped[kMaxFitParams + kPhi]   = x[3];
    } else if(nDim == 5){
      x_mapped[kXFrac]                 = x[0];
      x_mapped[kMNuNu]                 = x[1];
      x_mapped[kPhi]                   = x[2];
      x_mapped[kMaxFitParams + kXFrac] = x[3];
      x_mapped[kMaxFitParams + kMNuNu] = 0.;
      x_mapped[kMaxFitParams + kPhi]   = x[4];
    } else if(nDim == 6){
      x_mapped[kXFrac]                 = x[0];
      x_mapped[kMNuNu]                 = x[1];
      x_mapped[kPhi]                   = x[2];
      x_mapped[kMaxFitParams + kXFrac] = x[3];
      x_mapped[kMaxFitParams + kMNuNu] = x[4];
      x_mapped[kMaxFitParams + kPhi]   = x[5];
    } else assert(0);
    //std::cout << "<map_x>:" << std::endl;
    //for ( int i = 0; i < 6; ++i ) {
    //  std::cout << " x_mapped[" << i << "] = " << x_mapped[i] << std::endl;
    //}
  }
}

SVfitStandaloneAlgorithm::SVfitStandaloneAlgorithm(const std::vector<svFitStandalone::MeasuredTauLepton>& measuredTauLeptons, const svFitStandalone::Vector& measuredMET, const TMatrixD& covMET, 
						   unsigned int verbose) 
  : fitStatus_(-1), 
    verbose_(verbose), 
    maxObjFunctionCalls_(5000),
    mcObjectiveFunctionAdapter_(0),
    mcPtEtaPhiMassAdapter_(0),
    integrator2_(0),
    integrator2_nDim_(0),
    isInitialized2_(false),
    maxObjFunctionCalls2_(100000)
{ 
  // instantiate minuit, the arguments might turn into configurables once
  minimizer_ = ROOT::Math::Factory::CreateMinimizer("Minuit2", "Migrad");
  // instantiate the combined likelihood
  nll_ = new svFitStandalone::SVfitStandaloneLikelihood(measuredTauLeptons, measuredMET, covMET, (verbose_ > 2));
  nllStatus_ = nll_->error();

  clock_ = new TBenchmark();
}

SVfitStandaloneAlgorithm::~SVfitStandaloneAlgorithm() 
{
  delete nll_;
  delete minimizer_;
  delete mcObjectiveFunctionAdapter_;
  delete mcPtEtaPhiMassAdapter_;
  delete integrator2_;
}

void
SVfitStandaloneAlgorithm::setup()
{
  using namespace svFitStandalone;

  if ( verbose_ >= 1 ) {
    std::cout << "<SVfitStandaloneAlgorithm::setup()>:" << std::endl;
  }
  for ( size_t idx = 0; idx<nll_->measuredTauLeptons().size(); ++idx ) {
    if ( verbose_ >= 1 ){
      std::cout << " --> upper limit of leg1::mNuNu will be set to "; 
      if ( nll_->measuredTauLeptons()[idx].decayType() == kHadDecay ) { 
	std::cout << "0";
      } else {
	std::cout << (svFitStandalone::tauLeptonMass - TMath::Min(nll_->measuredTauLeptons()[idx].mass(), 1.5));
      } 
      std::cout << std::endl;
    }
    // start values for xFrac
    minimizer_->SetLimitedVariable(
      idx*kMaxFitParams + kXFrac, 
      std::string(TString::Format("leg%i::xFrac", (int)idx + 1)).c_str(), 
      0.5, 0.1, 0., 1.);
    // start values for nunuMass
    if ( nll_->measuredTauLeptons()[idx].decayType() == kHadDecay ) { 
      minimizer_->SetFixedVariable(
        idx*kMaxFitParams + kMNuNu, 
	std::string(TString::Format("leg%i::mNuNu", (int)idx + 1)).c_str(), 
	0.); 
    } else { 
      minimizer_->SetLimitedVariable(
        idx*kMaxFitParams + kMNuNu, 
	std::string(TString::Format("leg%i::mNuNu", (int)idx + 1)).c_str(), 
	0.8, 0.10, 0., svFitStandalone::tauLeptonMass - TMath::Min(nll_->measuredTauLeptons()[idx].mass(), 1.5)); 
    }
    // start values for phi
    minimizer_->SetVariable(
      idx*kMaxFitParams + kPhi, 
      std::string(TString::Format("leg%i::phi", (int)idx + 1)).c_str(), 
      0.0, 0.25);
  }
}

void
SVfitStandaloneAlgorithm::fit()
{
  if ( verbose_ >= 1 ) {
    std::cout << "<SVfitStandaloneAlgorithm::fit()>" << std::endl
	      << " dimension of fit    : " << nll_->measuredTauLeptons().size()*svFitStandalone::kMaxFitParams << std::endl
	      << " maxObjFunctionCalls : " << maxObjFunctionCalls_ << std::endl; 
  }
  // clear minimizer
  minimizer_->Clear();
  // set verbosity level of minimizer
  minimizer_->SetPrintLevel(-1);
  // setup the function to be called and the dimension of the fit
  ROOT::Math::Functor toMinimize(standaloneObjectiveFunctionAdapter_, nll_->measuredTauLeptons().size()*svFitStandalone::kMaxFitParams);
  minimizer_->SetFunction(toMinimize); 
  setup();
  minimizer_->SetMaxFunctionCalls(maxObjFunctionCalls_);
  // set Minuit strategy = 2, in order to get reliable error estimates:
  // http://www-cdf.fnal.gov/physics/statistics/recommendations/minuit.html
  minimizer_->SetStrategy(2);
  // compute uncertainties for increase of objective function by 0.5 wrt. 
  // minimum (objective function is log-likelihood function)
  minimizer_->SetErrorDef(0.5);
  if ( verbose_ >= 1 ) {
    std::cout << "starting ROOT::Math::Minimizer::Minimize..." << std::endl;
    std::cout << " #freeParameters = " << minimizer_->NFree() << ","
  	      << " #constrainedParameters = " << (minimizer_->NDim() - minimizer_->NFree()) << std::endl;
  }
  // do the minimization
  nll_->addDelta(false);
  nll_->addSinTheta(true);
  minimizer_->Minimize();
  if ( verbose_ >= 2 ) { 
    minimizer_->PrintResults(); 
  };
  /* get Minimizer status code, check if solution is valid:
    
     0: Valid solution
     1: Covariance matrix was made positive definite
     2: Hesse matrix is invalid
     3: Estimated distance to minimum (EDM) is above maximum
     4: Reached maximum number of function calls before reaching convergence
     5: Any other failure
  */
  fitStatus_ = minimizer_->Status();
  if ( verbose_ >=1 ) { 
    std::cout << "--> fitStatus = " << fitStatus_ << std::endl; 
  }
  
  // and write out the result
  using svFitStandalone::kXFrac;
  using svFitStandalone::kMNuNu;
  using svFitStandalone::kPhi;
  using svFitStandalone::kMaxFitParams;
  // update di-tau system with final fit results
  nll_->results(fittedTauLeptons_, minimizer_->X());
  // determine uncertainty of the fitted di-tau mass
  double x1RelErr = minimizer_->Errors()[kXFrac]/minimizer_->X()[kXFrac];
  double x2RelErr = minimizer_->Errors()[kMaxFitParams + kXFrac]/minimizer_->X()[kMaxFitParams + kXFrac];
  // this gives a unified treatment for retrieving the result for integration mode and fit mode
  fittedDiTauSystem_ = fittedTauLeptons_[0] + fittedTauLeptons_[1];
  mass_ = fittedDiTauSystem().mass();
  massUncert_ = TMath::Sqrt(0.25*x1RelErr*x1RelErr + 0.25*x2RelErr*x2RelErr)*fittedDiTauSystem().mass();
  if ( verbose_ >= 2 ) {
    std::cout << ">> -------------------------------------------------------------" << std::endl;
    std::cout << ">> Resonance Record: " << std::endl;
    std::cout << ">> -------------------------------------------------------------" << std::endl;
    std::cout << ">> pt  (di-tau)    = " << fittedDiTauSystem().pt  () << std::endl;
    std::cout << ">> eta (di-tau)    = " << fittedDiTauSystem().eta () << std::endl;
    std::cout << ">> phi (di-tau)    = " << fittedDiTauSystem().phi () << std::endl;
    std::cout << ">> mass(di-tau)    = " << fittedDiTauSystem().mass() << std::endl;  
    std::cout << ">> massUncert      = " << massUncert_ << std::endl
	      << "   error[xFrac1]   = " << minimizer_->Errors()[kXFrac] << std::endl
	      << "   value[xFrac1]   = " << minimizer_->X()[kXFrac]      << std::endl
	      << "   error[xFrac2]   = " << minimizer_->Errors()[kMaxFitParams+kXFrac] << std::endl
	      << "   value[xFrac2]   = " << minimizer_->X()[kMaxFitParams+kXFrac]      << std::endl;
    for ( size_t leg = 0; leg < 2 ; ++leg ){
      std::cout << ">> -------------------------------------------------------------" << std::endl;
      std::cout << ">> Leg " << leg+1 << " Record: " << std::endl;
      std::cout << ">> -------------------------------------------------------------" << std::endl;
      std::cout << ">> pt  (meas)      = " << nll_->measuredTauLeptons()[leg].p4().pt () << std::endl;
      std::cout << ">> eta (meas)      = " << nll_->measuredTauLeptons()[leg].p4().eta() << std::endl;
      std::cout << ">> phi (meas)      = " << nll_->measuredTauLeptons()[leg].p4().phi() << std::endl; 
      std::cout << ">> pt  (fit )      = " << fittedTauLeptons()[leg].pt () << std::endl;
      std::cout << ">> eta (fit )      = " << fittedTauLeptons()[leg].eta() << std::endl;
      std::cout << ">> phi (fit )      = " << fittedTauLeptons()[leg].phi() << std::endl; 
    }
  }
}

void
SVfitStandaloneAlgorithm::integrateVEGAS(const std::string& likelihoodFileName)
{
  using namespace svFitStandalone;
  
  if ( verbose_ >= 1 ){
    std::cout << "<SVfitStandaloneAlgorithm::integrateVEGAS>:" << std::endl;
    clock_->Start("<SVfitStandaloneAlgorithm::integrateVEGAS>");
  }

  double pi = 3.14159265;
  // number of hadrponic decays
  int khad = 0;
  for ( size_t idx = 0; idx < nll_->measuredTauLeptons().size(); ++idx ) {
    if ( nll_->measuredTauLeptons()[idx].decayType() == kHadDecay ) { 
      khad++; 
    }
  }
  // number of parameters for fit
  int par = nll_->measuredTauLeptons().size()*svFitStandalone::kMaxFitParams - (khad + 1);
  /* --------------------------------------------------------------------------------------
     lower and upper bounds for integration. Boundaries are deefined for each decay channel
     separately. The order is: 
     
     - 3dim : fully hadronic {xFrax, phihad1, phihad2}
     - 4dim : semi  leptonic {xFrac, nunuMass, philep, phihad}
     - 5dim : fully leptonic {xFrac, nunuMass1, philep1, nunuMass2, philep2}
     
     xl* defines the lower integation bound, xu* defines the upper integration bound in 
     the following definitions. 
     ATTENTION: order matters here! In the semi-leptonic decay the lepton must go first in 
     the parametrization, as it is first in the definition of integral boundaries. This is
     the reason why the measuredLeptons are eventually re-ordered in the constructor of 
     this class before passing them on to SVfitStandaloneLikelihood.
  */
  double xl3[3] = { 0.0, -pi, -pi };
  double xu3[3] = { 1.0,  pi,  pi };
  double xl4[4] = { 0.0, 0.0, -pi, -pi };
  double xu4[4] = { 1.0, svFitStandalone::tauLeptonMass, pi, pi };
  double xl5[5] = { 0.0, 0.0, -pi, 0.0, -pi };
  double xu5[5] = { 1.0, svFitStandalone::tauLeptonMass, pi, svFitStandalone::tauLeptonMass, pi };

  TH1* histogramMass = makeHistogram("SVfitStandaloneAlgorithm_histogramMass", measuredDiTauSystem().mass()/1.0125, 1.e+4, 1.025);
  TH1* histogramMass_density = (TH1*)histogramMass->Clone(Form("%s_density", histogramMass->GetName()));

  std::vector<double> xGraph;
  std::vector<double> xErrGraph;
  std::vector<double> yGraph;
  std::vector<double> yErrGraph;

  // integrator instance
  ROOT::Math::GSLMCIntegrator ig2("vegas", 0., 1.e-6, 10000);
  //ROOT::Math::GSLMCIntegrator ig2("vegas", 0., 1.e-6, 2000);
  ROOT::Math::Functor toIntegrate(&standaloneObjectiveFunctionAdapter_, &ObjectiveFunctionAdapter::Eval, par); 
  standaloneObjectiveFunctionAdapter_.SetPar(par);
  ig2.SetFunction(toIntegrate);
  nll_->addDelta(true);
  nll_->addSinTheta(false);
  nll_->addPhiPenalty(false);
  int count = 0;
  double pMax = 0.;
  double mtest = measuredDiTauSystem().mass();
  bool skiphighmasstail = false;
  for ( int i = 0; i < 100 && (!skiphighmasstail); ++i ) {
    standaloneObjectiveFunctionAdapter_.SetM(mtest);
    double p = -1.;
    if ( par == 4 ) {
      p = ig2.Integral(xl4, xu4);
    } else if ( par == 5 ) {
      p = ig2.Integral(xl5, xu5);
    } else if ( par == 3 ) {
      p = ig2.Integral(xl3, xu3);
    } else{
      std::cout << " >> ERROR : the nubmer of measured leptons must be 2" << std::endl;
      assert(0);
    }
    double pErr = ig2.Error();
    if ( verbose_ >= 2 ) {
      std::cout << "--> scan idx = " << i << ": mtest = " << mtest << ", p = " << p << " +/- " << pErr << " (pMax = " << pMax << ")" << std::endl;
    }    
    if ( p > pMax ) {
      mass_ = mtest;
      pMax  = p;
      count = 0;
    } else {
      if ( p < (1.e-3*pMax) ) {
	++count;
	if ( count>= 5 ) {
	  skiphighmasstail = true;
	}
      } else {
	count = 0;
      }
    }
    double mtest_step = TMath::Max(2.5, 0.025*mtest);
    int bin = histogramMass->FindBin(mtest);
    histogramMass->SetBinContent(bin, p*mtest_step);
    histogramMass->SetBinError(bin, pErr*mtest_step);
    xGraph.push_back(mtest);
    xErrGraph.push_back(0.5*mtest_step);
    yGraph.push_back(p);
    yErrGraph.push_back(pErr);
    mtest += mtest_step;
  }
  //mass_ = extractValue(histogramMass, histogramMass_density);
  massUncert_ = extractUncertainty(histogramMass, histogramMass_density);
  if ( verbose_ >= 1 ) {
    std::cout << "--> mass  = " << mass_  << " +/- " << massUncert_ << std::endl;
    std::cout << "   (pMax = " << pMax << ", count = " << count << ")" << std::endl;
  }
  delete histogramMass;
  delete histogramMass_density;
  if ( likelihoodFileName != "" ) {
    size_t numPoints = xGraph.size();
    TGraphErrors* likelihoodGraph = new TGraphErrors(numPoints);
    likelihoodGraph->SetName("svFitLikelihoodGraph");
    for ( size_t iPoint = 0; iPoint < numPoints; ++iPoint ) {
      likelihoodGraph->SetPoint(iPoint, xGraph[iPoint], yGraph[iPoint]);
      likelihoodGraph->SetPointError(iPoint, xErrGraph[iPoint], yErrGraph[iPoint]);
    }
    TFile* likelihoodFile = new TFile(likelihoodFileName.data(), "RECREATE");
    likelihoodGraph->Write();
    delete likelihoodFile;
    delete likelihoodGraph;
  }

  if ( verbose_ >= 1 ) {
    clock_->Show("<SVfitStandaloneAlgorithm::integrateVEGAS>");
  }
}

void
SVfitStandaloneAlgorithm::integrateMarkovChain()
{
  using namespace svFitStandalone;
  
  if ( verbose_ >= 1 ) {
    std::cout << "<SVfitStandaloneAlgorithm::integrateMarkovChain>:" << std::endl;
    clock_->Start("<SVfitStandaloneAlgorithm::integrateMarkovChain>");
  }
  if ( isInitialized2_ ) {
    mcPtEtaPhiMassAdapter_->Reset();
  } else {
    // initialize    
    std::string initMode = "none";
    unsigned numIterBurnin = TMath::Nint(0.10*maxObjFunctionCalls2_);
    unsigned numIterSampling = maxObjFunctionCalls2_;
    unsigned numIterSimAnnealingPhase1 = TMath::Nint(0.02*maxObjFunctionCalls2_);
    unsigned numIterSimAnnealingPhase2 = TMath::Nint(0.06*maxObjFunctionCalls2_);
    double T0 = 15.;
    double alpha = 1.0 - 1.e+2/maxObjFunctionCalls2_;
    unsigned numChains = 1;
    unsigned numBatches = 1;
    unsigned L = 1;
    double epsilon0 = 1.e-2;
    double nu = 0.71;
    int verbose = -1;
    integrator2_ = new SVfitStandaloneMarkovChainIntegrator(
                         initMode, numIterBurnin, numIterSampling, numIterSimAnnealingPhase1, numIterSimAnnealingPhase2,
			 T0, alpha, numChains, numBatches, L, epsilon0, nu,
			 verbose);
    mcObjectiveFunctionAdapter_ = new MCObjectiveFunctionAdapter();
    integrator2_->setIntegrand(*mcObjectiveFunctionAdapter_);
    integrator2_nDim_ = 0;
    mcPtEtaPhiMassAdapter_ = new MCPtEtaPhiMassAdapter();
    integrator2_->registerCallBackFunction(*mcPtEtaPhiMassAdapter_);
    isInitialized2_ = true;    
  }

  const double pi = TMath::Pi();
  // number of hadronic decays
  int khad = 0;
  for ( size_t idx = 0; idx < nll_->measuredTauLeptons().size(); ++idx ) {
    if ( nll_->measuredTauLeptons()[idx].decayType() == kHadDecay ) { 
      ++khad; 
    }
  }
  // number of parameters for fit
  int nDim = nll_->measuredTauLeptons().size()*svFitStandalone::kMaxFitParams - khad;  
  if ( nDim != integrator2_nDim_ ) {
    mcObjectiveFunctionAdapter_->SetNDim(nDim);
    integrator2_->setIntegrand(*mcObjectiveFunctionAdapter_);
    mcPtEtaPhiMassAdapter_->SetNDim(nDim);
    integrator2_nDim_ = nDim;
  }
  /* --------------------------------------------------------------------------------------
     lower and upper bounds for integration. Boundaries are defined for each decay channel
     separately. The order is: 
     
     - 4dim : fully hadronic {xhad1, phihad1, xhad2, phihad2}
     - 5dim : semi  leptonic {xlep, nunuMass, philep, xhad, phihad}
     - 6dim : fully leptonic {xlep1, nunuMass1, philep1, xlep2, nunuMass2, philep2}
     
     x0* defines the start value for the integration, xl* defines the lower integation bound, 
     xu* defines the upper integration bound in the following definitions. 
     ATTENTION: order matters here! In the semi-leptonic decay the lepton must go first in 
     the parametrization, as it is first in the definition of integral boundaries. This is
     the reason why the measuredLeptons are eventually re-ordered in the constructor of 
     this class before passing them on to SVfitStandaloneLikelihood.
  */
  double x04[4] = { 0.5, 0.0, 0.5, 0.0 };
  double xl4[4] = { 0.0, -pi, 0.0, -pi };
  double xu4[4] = { 1.0,  pi, 1.0,  pi };
  double x05[5] = { 0.5, 0.8, 0.0, 0.5, 0.0 };
  double xl5[5] = { 0.0, 0.0, -pi, 0.0, -pi };
  double xu5[5] = { 1.0, svFitStandalone::tauLeptonMass, pi, 1.0, pi };
  xu5[1] = svFitStandalone::tauLeptonMass - TMath::Min(nll_->measuredTauLeptons()[0].mass(), 1.6);
  x05[1] = 0.5*(xl5[1] + xu5[1]);
  double x06[6] = { 0.5, 0.8, 0.0, 0.5, 0.8, 0.0 };
  double xl6[6] = { 0.0, 0.0, -pi, 0.0, 0.0, -pi };
  double xu6[6] = { 1.0, svFitStandalone::tauLeptonMass, pi, 1.0, svFitStandalone::tauLeptonMass, pi };
  xu6[1] = svFitStandalone::tauLeptonMass - TMath::Min(nll_->measuredTauLeptons()[0].mass(), 1.6);
  x06[1] = 0.5*(xl6[1] + xu6[1]);
  xu6[4] = svFitStandalone::tauLeptonMass - TMath::Min(nll_->measuredTauLeptons()[1].mass(), 1.6);
  x06[4] = 0.5*(xl6[4] + xu6[4]);
  std::vector<double> x0(nDim);
  std::vector<double> xl(nDim);
  std::vector<double> xu(nDim);
  for ( int i = 0; i < nDim; ++i ) {
    if ( nDim == 4 ){
      x0[i] = x04[i];
      xl[i] = xl4[i];
      xu[i] = xu4[i];
    } else if ( nDim == 5 ) {
      x0[i] = x05[i];
      xl[i] = xl5[i];
      xu[i] = xu5[i];
    } else if ( nDim == 6 ) {
      x0[i] = x06[i];
      xl[i] = xl6[i];
      xu[i] = xu6[i];
    } else {
      std::cerr << "<SVfitStandaloneAlgorithm>:"
		<< "Exactly 2 measured leptons required --> ABORTING !!\n";
      assert(0);
    }
    // transform startPosition into interval ]0..1[
    // expected by MarkovChainIntegrator class
    x0[i] = (x0[i] - xl[i])/(xu[i] - xl[i]);
    //std::cout << "x0[" << i << "] = " << x0[i] << std::endl;
  }
  integrator2_->initializeStartPosition_and_Momentum(x0);
  nll_->addDelta(false);
  nll_->addSinTheta(false);
  nll_->addPhiPenalty(false);
  double integral = 0.;
  double integralErr = 0.;
  int errorFlag = 0;
  integrator2_->integrate(xl, xu, integral, integralErr, errorFlag);
  fitStatus_ = errorFlag;
  pt_ = mcPtEtaPhiMassAdapter_->getPt();
  ptUncert_ = mcPtEtaPhiMassAdapter_->getPtUncert();
  eta_ = mcPtEtaPhiMassAdapter_->getEta();
  etaUncert_ = mcPtEtaPhiMassAdapter_->getEtaUncert();
  phi_ = mcPtEtaPhiMassAdapter_->getPhi();
  phiUncert_ = mcPtEtaPhiMassAdapter_->getPhiUncert();
  mass_ = mcPtEtaPhiMassAdapter_->getMass();
  massUncert_ = mcPtEtaPhiMassAdapter_->getMassUncert();
  fittedDiTauSystem_ = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double> >(pt_, eta_, phi_, mass_);
  if ( verbose_ >= 1 ) {
    std::cout << "--> Pt = " << pt_ << ", eta = " << eta_ << ", phi = " << phi_ << ", mass  = " << mass_ << std::endl;
    clock_->Show("<SVfitStandaloneAlgorithm::integrateMarkovChain>");
  }
}
