#include "TauAnalysis/SVfitStandalone/interface/SVfitStandaloneLikelihood.h"

#include "TauAnalysis/SVfitStandalone/interface/svFitStandaloneAuxFunctions.h"
#include "TauAnalysis/SVfitStandalone/interface/LikelihoodFunctions.h"

using namespace svFitStandalone;

/// global function pointer for minuit or VEGAS
const SVfitStandaloneLikelihood* SVfitStandaloneLikelihood::gSVfitStandaloneLikelihood = 0;
/// indicate first iteration for integration or fit cycle for debugging
static bool FIRST = true;

SVfitStandaloneLikelihood::SVfitStandaloneLikelihood(const std::vector<MeasuredTauLepton>& measuredTauLeptons, const Vector& measuredMET, const TMatrixD& covMET, bool verbose) 
  : metPower_(1.0), 
    addLogM_(false), 
    addDelta_(true),
    addSinTheta_(false),
    addPhiPenalty_(true),
    verbose_(verbose), 
    idxObjFunctionCall_(0), 
    invCovMET_(2,2),
    errorCode_(0),
    shiftVisMassAndPt_(false),
    l1lutVisMassRes_(0),
    l1lutVisPtRes_(0),
    l2lutVisMassRes_(0),
    l2lutVisPtRes_(0)
{
  if ( verbose_ ) {
    std::cout << "<SVfitStandaloneLikelihood::SVfitStandaloneLikelihood>:" << std::endl;
  }
  measuredMET_ = measuredMET;
  // for integration mode the order of lepton or tau matters due to the choice of order in which 
  // way the integration boundaries are defined. In this case the lepton should always go before
  // the tau. For tau-tau or lep-lep the order is irrelevant.
  if ( measuredTauLeptons[0].type() == svFitStandalone::kHadDecay ) {
    measuredTauLeptons_.push_back(measuredTauLeptons[1]);
    measuredTauLeptons_.push_back(measuredTauLeptons[0]);
  } else {
    measuredTauLeptons_= measuredTauLeptons;
  }
  if ( measuredTauLeptons_.size() != 2 ) {
    std::cout << " >> ERROR : the number of measured leptons must be 2 but is found to be: " << measuredTauLeptons_.size() << std::endl;
    errorCode_ |= LeptonNumber;
  }
  // determine transfer matrix for MET
  invCovMET_= covMET;
  covDet_ = invCovMET_.Determinant();
  if ( covDet_ != 0 ) { 
    invCovMET_.Invert(); 
  } else{
    std::cout << " >> ERROR: cannot invert MET covariance Matrix (det=0)." << std::endl;
    errorCode_ |= MatrixInversion;
  }
  // set global function pointer to this
  gSVfitStandaloneLikelihood = this;
}

void 
SVfitStandaloneLikelihood::shiftVisMassAndPt(bool value, const TH1* l1lutVisMassRes, const TH1* l1lutVisPtRes, const TH1* l2lutVisMassRes, const TH1* l2lutVisPtRes)
{
  shiftVisMassAndPt_ = value;
  if ( shiftVisMassAndPt_ ) {
    l1lutVisMassRes_ = l1lutVisMassRes;
    l1lutVisPtRes_   = l1lutVisPtRes;
    l2lutVisMassRes_ = l2lutVisMassRes;
    l2lutVisPtRes_   = l2lutVisPtRes;
  }
}

const double*
SVfitStandaloneLikelihood::transform(double* xPrime, const double* x) const
{
  if ( verbose_ ) {
    std::cout << "<SVfitStandaloneLikelihood:transform(double*, const double*)>:" << std::endl;
  }
  LorentzVector fittedDiTauSystem;
  for ( size_t idx = 0; idx < measuredTauLeptons_.size(); ++idx ) {
    const MeasuredTauLepton& measuredTauLepton = measuredTauLeptons_[idx];

    // map to local variables to be more clear on the meaning of the individual parameters. The fit parameters are ayered 
    // for each tau decay
    double nunuMass, labframeXFrac, labframePhi;
    double visMass_unshifted = measuredTauLepton.mass();
    double visMass = visMass_unshifted; // visible momentum in lab-frame
    double labframeVisMom_unshifted = measuredTauLepton.momentum(); 
    double labframeVisMom = labframeVisMom_unshifted; // visible momentum in lab-frame
    double labframeVisEn  = measuredTauLepton.energy(); // visible energy in lab-frame    
    if ( measuredTauLepton.type() == kLepDecay ) {
      labframeXFrac = x[idx*kMaxFitParams + kXFrac];
      nunuMass = x[idx*kMaxFitParams + kMNuNu];
      labframePhi = x[idx*kMaxFitParams + kPhi];
    } else {
      labframeXFrac = x[idx*kMaxFitParams + kXFrac];
      nunuMass = 0.;
      labframePhi = x[idx*kMaxFitParams + kPhi];
      if ( shiftVisMassAndPt_ ) {
	visMass = x[idx*kMaxFitParams + kVisMassShifted];
	labframeVisMom *= (1. + x[idx*kMaxFitParams + kRecTauPtDivGenTauPt]);
	labframeVisEn = TMath::Sqrt(labframeVisMom*labframeVisMom + visMass*visMass);
      }
    }
    // add protection against unphysical visible energy fractions:
    // return 0 pointer will lead to 0 evaluation of prob
    if ( !(labframeXFrac >= 0. && labframeXFrac <= 1.) ) {
      return 0;
    }
    // add protection against zero mass for visMass. If visMass is lower than the electron mass, set it
    // to the electron mass
    if ( visMass < 5.1e-4 ) { 
      visMass = 5.1e-4; 
    }    
    // momentum of visible decay products in tau lepton restframe
    double restframeVisMom     = pVisRestFrame(visMass, nunuMass, tauLeptonMass);
    // tau lepton decay angle in tau lepton restframe (as function of the energy ratio of visible decay products/tau lepton energy)
    double restframeDecayAngle = gjAngleFromX(labframeXFrac, visMass, restframeVisMom, labframeVisEn, tauLeptonMass);
    // tau lepton decay angle in labframe
    double labframeDecayAngle  = gjAngleToLabFrame(restframeVisMom, restframeDecayAngle, labframeVisMom);
    // tau lepton momentum in labframe
    double labframeTauMom      = motherMomentumLabFrame(visMass, restframeVisMom, restframeDecayAngle, labframeVisMom, tauLeptonMass);
    Vector labframeTauDir      = motherDirection(measuredTauLeptons_[idx].direction(), labframeDecayAngle, labframePhi).unit();
    // tau lepton four vector in labframe
    fittedDiTauSystem += motherP4(labframeTauDir, labframeTauMom, tauLeptonMass);
    // fill branch-wise nll parameters
    xPrime[ idx == 0 ? kNuNuMass1            : kNuNuMass2            ] = nunuMass;
    xPrime[ idx == 0 ? kVisMass1             : kVisMass2             ] = visMass;
    xPrime[ idx == 0 ? kDecayAngle1          : kDecayAngle2          ] = restframeDecayAngle;
    xPrime[ idx == 0 ? kDeltaVisMass1        : kDeltaVisMass2        ] = visMass_unshifted - visMass;
    xPrime[ idx == 0 ? kRecTauPtDivGenTauPt1 : kRecTauPtDivGenTauPt2 ] = ( labframeVisMom > 0. ) ? (labframeVisMom_unshifted/labframeVisMom) : 1.e+3;
    xPrime[ idx == 0 ? kMaxNLLParams         : (kMaxNLLParams + 1)   ] = labframeXFrac;
  }
 
  Vector fittedMET = fittedDiTauSystem.Vect() - (measuredTauLeptons_[0].p()+measuredTauLeptons_[1].p()); 
  // fill event-wise nll parameters
  xPrime[ kDMETx   ] = measuredMET_.x() - fittedMET.x(); 
  xPrime[ kDMETy   ] = measuredMET_.y() - fittedMET.y();
  xPrime[ kMTauTau ] = fittedDiTauSystem.mass();

  if ( verbose_ && FIRST ) {
    std::cout << " >> input values for transformed variables: " << std::endl;
    std::cout << "    MET[x] = " <<  fittedMET.x() << " (fitted)  " << measuredMET_.x() << " (measured) " << std::endl; 
    std::cout << "    MET[y] = " <<  fittedMET.y() << " (fitted)  " << measuredMET_.y() << " (measured) " << std::endl; 
    std::cout << "    fittedDiTauSystem: [" 
	      << " px = " << fittedDiTauSystem.px() 
	      << " py = " << fittedDiTauSystem.py() 
	      << " pz = " << fittedDiTauSystem.pz() 
	      << " En = " << fittedDiTauSystem.energy() 
	      << " ]" << std::endl; 
    std::cout << " >> nll parameters after transformation: " << std::endl;
    std::cout << "    x[kNuNuMass1  ] = " << xPrime[kNuNuMass1  ] << std::endl;
    std::cout << "    x[kVisMass1   ] = " << xPrime[kVisMass1   ] << std::endl;
    std::cout << "    x[kDecayAngle1] = " << xPrime[kDecayAngle1] << std::endl;
    std::cout << "    x[kNuNuMass2  ] = " << xPrime[kNuNuMass2  ] << std::endl;
    std::cout << "    x[kVisMass2   ] = " << xPrime[kVisMass2   ] << std::endl;
    std::cout << "    x[kDecayAngle2] = " << xPrime[kDecayAngle2] << std::endl;
    std::cout << "    x[kDMETx      ] = " << xPrime[kDMETx      ] << std::endl;
    std::cout << "    x[kDMETy      ] = " << xPrime[kDMETy      ] << std::endl;
    std::cout << "    x[kMTauTau    ] = " << xPrime[kMTauTau    ] << std::endl;
  }
  return xPrime;
}

double
SVfitStandaloneLikelihood::prob(const double* x) const 
{
  // in case of initialization errors don't start to do anything
  if ( error() ) { 
    return 0.;
  }
  if ( verbose_ ) {
    std::cout << "<SVfitStandaloneLikelihood:prob(const double*)>:" << std::endl;
  }
  ++idxObjFunctionCall_;
  if ( verbose_ && FIRST ) {
    std::cout << " >> ixdObjFunctionCall : " << idxObjFunctionCall_ << std::endl;  
  }
  // prevent kPhi in the fit parameters (kFitParams) from trespassing the 
  // +/-pi boundaries
  double phiPenalty = 0.;
  if ( addPhiPenalty_ ) {
    for ( size_t idx = 0; idx < measuredTauLeptons_.size(); ++idx ) {
      if ( TMath::Abs(idx*kMaxFitParams + x[kPhi]) > TMath::Pi() ) {
	phiPenalty += (TMath::Abs(x[kPhi]) - TMath::Pi())*(TMath::Abs(x[kPhi]) - TMath::Pi());
      }
    }
  }
  // xPrime are the transformed variables from which to construct the nll
  // transform performs the transformation from the fit parameters x to the 
  // nll parameters xPrime. prob is the actual combined likelihood. The
  // phiPenalty prevents the fit to converge to unphysical values beyond
  // +/-pi 
  double xPrime[kMaxNLLParams + 2];
  const double* xPrime_ptr = transform(xPrime, x);
  if ( xPrime_ptr ) {
    return prob(xPrime_ptr, phiPenalty);
  } else {
    return 0.;
  }
}

double 
SVfitStandaloneLikelihood::prob(const double* xPrime, double phiPenalty) const
{
  if ( verbose_ && FIRST ) {
    std::cout << "<SVfitStandaloneLikelihood:prob(const double*, double)>:" << std::endl;
  }
  // start the combined likelihood construction from MET
  double prob = probMET(xPrime[kDMETx], xPrime[kDMETy], covDet_, invCovMET_, metPower_, (verbose_&& FIRST));
  if ( verbose_ && FIRST ) {
    std::cout << "probMET         = " << prob << std::endl;
  }
  // add likelihoods for the decay branches
  for ( size_t idx = 0; idx < measuredTauLeptons_.size(); ++idx ) {
    switch ( measuredTauLeptons_[idx].type() ) {
    case kHadDecay :
      prob *= probTauToHadPhaseSpace(
                xPrime[idx == 0 ? kDecayAngle1 : kDecayAngle2], 
		xPrime[idx == 0 ? kNuNuMass1 : kNuNuMass2], 
		xPrime[idx == 0 ? kVisMass1 : kVisMass2], 
		xPrime[idx == 0 ? kMaxNLLParams : (kMaxNLLParams + 1)], 
		addSinTheta_, 
		(verbose_&& FIRST));
      if ( shiftVisMassAndPt_ ) {
	prob *= probVisMassAndPtShift(
                  xPrime[idx == 0 ? kDeltaVisMass1 : kDeltaVisMass2], 
		  xPrime[idx == 0 ? kRecTauPtDivGenTauPt1 : kRecTauPtDivGenTauPt2], 
		  l1lutVisMassRes_, l1lutVisPtRes_);
      }
      if ( verbose_ && FIRST ) {
	std::cout << " *probTauToHad  = " << prob << std::endl;
      }
      break;
    case kLepDecay :
      prob *= probTauToLepMatrixElement(
		xPrime[idx == 0 ? kDecayAngle1 : kDecayAngle2], 
		xPrime[idx == 0 ? kNuNuMass1 : kNuNuMass2], 
		xPrime[idx == 0 ? kVisMass1 : kVisMass2], 
		xPrime[idx == 0 ? kMaxNLLParams : (kMaxNLLParams + 1)], 
		addSinTheta_, 
		(verbose_&& FIRST));
      if ( verbose_ && FIRST ) {
	std::cout << " *probTauToLep  = " << prob << std::endl;
      }
      break;
    }
  }
  // add additional logM term if configured such 
  if ( addLogM_ ) {
    if ( xPrime[kMTauTau] > 0. ) {
      prob *= (1.0/xPrime[kMTauTau]);
    }
    if ( verbose_ && FIRST ) {
      std::cout << " *1/mtautau     = " << prob << std::endl;
    }
  }
  if ( addDelta_ ) {
    prob *= (2.*xPrime[kMaxNLLParams + 1]/xPrime[kMTauTau]);
    if ( verbose_ && FIRST ) {
      std::cout << " *deltaDeriv.   = " << prob << std::endl;
    }
  }
  // add additional phiPenalty in case kPhi in the fit parameters 
  // (kFitParams) trespassed the physical boundaries from +/-pi 
  if ( phiPenalty > 0. ) {
    prob *= TMath::Exp(-phiPenalty);
    if ( verbose_ && FIRST ) {
      std::cout << "* phiPenalty   = " << prob << std::endl;
    }
  }
  // set FIRST to false after the first complete evaluation of the likelihood 
  FIRST = false;
  return prob;
}

void
SVfitStandaloneLikelihood::results(std::vector<LorentzVector>& fittedTauLeptons, const double* x) const
{
  if ( verbose_ ) {
    std::cout << "<SVfitStandaloneLikelihood:results(std::vector<LorentzVector>&, const double*)>:" << std::endl;
  }
  for ( size_t idx = 0; idx < measuredTauLeptons_.size(); ++idx ) {
    const MeasuredTauLepton& measuredTauLepton = measuredTauLeptons_[idx];

    // map to local variables to be more clear on the meaning of the individual parameters. The fit parameters are ayered 
    // for each tau decay
    double nunuMass                 = x[ idx*kMaxFitParams + kMNuNu ];       // nunu inv mass (can be const 0 for had tau decays) 
    double labframeXFrac            = x[ idx*kMaxFitParams + kXFrac ];       // visible energy fraction x in labframe
    double labframePhi              = x[ idx*kMaxFitParams + kPhi   ];       // phi in labframe 
    double visMass                  = measuredTauLepton.mass(); 
    double labframeVisMom_unshifted = measuredTauLepton.momentum(); 
    double labframeVisMom           = labframeVisMom_unshifted; // visible momentum in lab-frame
    double labframeVisEn            = measuredTauLepton.energy(); // visible energy in lab-frame    
    if ( measuredTauLepton.type() == kHadDecay && shiftVisMassAndPt_ ) {
      visMass = x[ idx*kMaxFitParams + kVisMassShifted ];
      labframeVisMom *= (1. + x[ idx*kMaxFitParams + kRecTauPtDivGenTauPt ]);
      labframeVisEn = TMath::Sqrt(labframeVisMom*labframeVisMom + visMass*visMass);
    }
    // momentum of visible decay products in tau lepton restframe
    double restframeVisMom     = pVisRestFrame(visMass, nunuMass, tauLeptonMass);
    // tau lepton decay angle in tau lepton restframe (as function of the energy ratio of visible decay products/tau lepton energy)
    double restframeDecayAngle = gjAngleFromX(labframeXFrac, visMass, restframeVisMom, labframeVisEn, tauLeptonMass);
    // tau lepton decay angle in labframe
    double labframeDecayAngle  = gjAngleToLabFrame(restframeVisMom, restframeDecayAngle, labframeVisMom);
    // tau lepton momentum in labframe
    double labframeTauMom      = motherMomentumLabFrame(visMass, restframeVisMom, restframeDecayAngle, labframeVisMom, tauLeptonMass);
    Vector labframeTauDir      = motherDirection(measuredTauLepton.direction(), labframeDecayAngle, labframePhi).unit();
    // tau lepton four vector in labframe
    if ( idx < fittedTauLeptons.size() ) fittedTauLeptons[idx] = motherP4(labframeTauDir, labframeTauMom, tauLeptonMass);
    else fittedTauLeptons.push_back(motherP4(labframeTauDir, labframeTauMom, tauLeptonMass));
  }
}
