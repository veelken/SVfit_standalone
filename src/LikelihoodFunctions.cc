#include "TauAnalysis/SVfitStandalone/interface/LikelihoodFunctions.h"

#include "TauAnalysis/SVfitStandalone/interface/svFitStandaloneAuxFunctions.h"

#include <TMath.h>

#include <iostream>

using namespace svFitStandalone;

double 
probMET(double dMETX, double dMETY, double covDet, const TMatrixD& covInv, double power, bool verbose)
{
  if ( verbose ) {
    std::cout << "<probMET>:" << std::endl;
    std::cout << " dMETX = " << dMETX << std::endl;
    std::cout << " dMETY = " << dMETY << std::endl;
    std::cout << " covDet = " << covDet << std::endl;
    std::cout << " covInv:" << std::endl;
    covInv.Print();
  }
  double nll = 0.;
  if ( covDet != 0. ) {
    nll = TMath::Log(2.*TMath::Pi()) + 0.5*TMath::Log(TMath::Abs(covDet)) 
         + 0.5*(dMETX*(covInv(0,0)*dMETX + covInv(0,1)*dMETY) + dMETY*(covInv(1,0)*dMETX + covInv(1,1)*dMETY));
  } else {
    nll = std::numeric_limits<float>::max();
  }
  double prob = TMath::Exp(-power*nll);
  if ( verbose ) {
    std::cout << "--> prob = " << prob << std::endl;
  }
  return prob; 
}

double 
probTauToLepMatrixElement(double decayAngle, double nunuMass, double visMass, double x, bool applySinTheta, bool verbose)
{
  if ( verbose ) {
    std::cout << "<probTauToLepMatrixElement>:" << std::endl;
    std::cout << " decayAngle = " << decayAngle << std::endl;
    std::cout << " nunuMass = " << nunuMass << std::endl;
    std::cout << " visMass = " << visMass << std::endl;
    std::cout << " x = " << x << std::endl;
    std::cout << " applySinTheta = " << applySinTheta << std::endl;
  }
  double nuMass2 = nunuMass*nunuMass;
  // protect against rounding errors that may lead to negative masses
  if ( nunuMass < 0. ) nunuMass = 0.; 
  double prob = 0.;
  if ( nunuMass < TMath::Sqrt((1. - x)*tauLeptonMass2) ) { // LB: physical solution
    prob = (13./tauLeptonMass4)*(tauLeptonMass2 - nuMass2)*(tauLeptonMass2 + 2.*nuMass2)*nunuMass;
  } else {    
    double nunuMass_limit  = TMath::Sqrt((1. - x)*tauLeptonMass2);
    double nunuMass2_limit = nunuMass_limit*nunuMass_limit;
    prob = (13./tauLeptonMass4)*(tauLeptonMass2 - nunuMass2_limit)*(tauLeptonMass2 + 2.*nunuMass2_limit)*nunuMass_limit;
    prob /= (1. + 1.e+6*TMath::Power(nunuMass - nunuMass_limit, 2.));
  }
  if ( applySinTheta ) prob *= (0.5*TMath::Sin(decayAngle));
  if ( verbose ) {
    std::cout << "--> prob = " << prob << std::endl;
  }
  return prob;
}

double 
probTauToHadPhaseSpace(double decayAngle, double nunuMass, double visMass, double x, bool applySinTheta, bool verbose)
{
  if ( verbose ) {
    std::cout << "<probTauToHadPhaseSpace>:" << std::endl;
    std::cout << " decayAngle = " << decayAngle << std::endl;
    std::cout << " nunuMass = " << nunuMass << std::endl;
    std::cout << " visMass = " << visMass << std::endl;
    std::cout << " x = " << x << std::endl;
    std::cout << " applySinTheta = " << applySinTheta << std::endl;
  }
  double Pvis_rf = svFitStandalone::pVisRestFrame(visMass, nunuMass, svFitStandalone::tauLeptonMass);
  double visMass2 = visMass*visMass;
  double prob = tauLeptonMass/(2.*Pvis_rf);
  if ( x < (visMass2/tauLeptonMass2) ) {
    double x_limit = visMass2/tauLeptonMass2;
    prob /= (1. + 1.e+6*TMath::Power(x - x_limit, 2.));
  } else if ( x > 1. ) {
    double visEnFracX_limit = 1.;
    prob /= (1. + 1.e+6*TMath::Power(x - visEnFracX_limit, 2.));
  }
  if ( applySinTheta ) prob *= (0.5*TMath::Sin(decayAngle));
  if ( verbose ) {
    std::cout << "--> prob = " << prob << std::endl;
  }
  return prob;
}

namespace
{
  double extractProbFromLUT(double x, const TH1* lut)
  {
    int bin = (const_cast<TH1*>(lut))->FindBin(x);
    int numBins = lut->GetNbinsX();
    if ( bin < 1       ) bin = 1;
    if ( bin > numBins ) bin = numBins;
    return lut->GetBinContent(bin);
  }
}

double 
probVisMassAndPtShift(double deltaVisMass, double recTauPtDivGenTauPt, const TH1* lutVisMassRes, const TH1* lutVisPtRes)
{
  double prob = extractProbFromLUT(deltaVisMass, lutVisMassRes);
  prob *= extractProbFromLUT(recTauPtDivGenTauPt, lutVisPtRes);
  return prob;
}
