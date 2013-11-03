#include "TauAnalysis/SVfitStandalone/interface/svFitStandaloneAuxFunctions.h"

#include <TMath.h>

namespace svFitStandalone
{
  //-----------------------------------------------------------------------------
  // define auxiliary functions for internal usage
  inline double energyFromMomentum(double momentum, double mass) {
    return TMath::Sqrt(square(mass) + square(momentum));
  }
  
  // Adapted for our vector types from TVector3 class
  Vector rotateUz(const ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> >& toRotate, const Vector& newUzVector)
  {
    // NB: newUzVector must be a unit vector !
    Double_t u1 = newUzVector.X();
    Double_t u2 = newUzVector.Y();
    Double_t u3 = newUzVector.Z();
    Double_t up = u1*u1 + u2*u2;

    Double_t fX = toRotate.X();
    Double_t fY = toRotate.Y();
    Double_t fZ = toRotate.Z();

    if ( up ) {
      up = TMath::Sqrt(up);
      Double_t px = fX;
      Double_t py = fY;
      Double_t pz = fZ;
      fX = (u1*u3*px - u2*py + u1*up*pz)/up;
      fY = (u2*u3*px + u1*py + u2*up*pz)/up;
      fZ = (u3*u3*px -    px + u3*up*pz)/up;
    } else if ( u3 < 0. ) {
      fX = -fX;
      fZ = -fZ;
    } else {}; // phi = 0, theta = pi

    return Vector(fX, fY, fZ);
  }

  double getMeanOfBinsAboveThreshold(const TH1* histogram, double threshold, int verbosity)
  {
    //std::cout << "<getMeanOfBinsAboveThreshold>:" << std::endl;
    //std::cout << " threshold = " << threshold << std::endl;
    
    double mean = 0.;
    double normalization = 0.;
    int numBins = histogram->GetNbinsX();
    for ( int iBin = 1; iBin <= numBins; ++iBin ) {
      double binCenter = histogram->GetBinCenter(iBin);
      double binContent = histogram->GetBinContent(iBin);
      if ( binContent >= threshold ) {
        if ( verbosity ) std::cout << " adding binContent = " << binContent << " @ binCenter = " << binCenter << std::endl;
        mean += (binCenter*binContent);
        normalization += binContent;
      }
    }
    if ( normalization > 0. ) mean /= normalization;
    if ( verbosity ) std::cout << "--> mean = " << mean << std::endl;
    return mean;
  }
  //-----------------------------------------------------------------------------

  double gjAngleFromX(double x, double visMass, double pVis_rf, double enVis_lab, double motherMass) 
  {
    double enVis_rf = energyFromMomentum(pVis_rf, visMass);
    double beta = TMath::Sqrt(1. - square(motherMass*x/enVis_lab));
    double cosGjAngle = (motherMass*x - enVis_rf)/(pVis_rf*beta);
    double gjAngle = TMath::ACos(cosGjAngle);
    return gjAngle;
  }

  double pVisRestFrame(double visMass, double invisMass, double motherMass)
  {
    double motherMass2 = motherMass*motherMass;
    double pVis = TMath::Sqrt((motherMass2 - square(visMass + invisMass))
                             *(motherMass2 - square(visMass - invisMass)))/(2.*motherMass);
    return pVis;
  }

  double gjAngleToLabFrame(double pVisRestFrame, double gjAngle, double pVisLabFrame)
  {
    // Get the compenent of the rest frame momentum perpindicular to the tau
    // boost direction. This quantity is Lorentz invariant.
    double pVisRestFramePerp = pVisRestFrame*TMath::Sin(gjAngle);

    // Determine the corresponding opening angle in the LAB frame
    double gjAngleLabFrame = TMath::ASin(pVisRestFramePerp/pVisLabFrame);

    return gjAngleLabFrame;
  }

  double motherMomentumLabFrame(double visMass, double pVisRestFrame, double gjAngle, double pVisLabFrame, double motherMass)
  {
    // Determine the corresponding opening angle in the LAB frame
    double angleVisLabFrame = gjAngleToLabFrame(pVisRestFrame, gjAngle, pVisLabFrame);

    // Get the visible momentum perpindicular/parallel to the tau boost direction in the LAB
    double pVisLabFrame_parallel = pVisLabFrame*TMath::Cos(angleVisLabFrame);

    // Now use the Lorentz equation for pVis along the tau direction to solve for
    // the gamma of the tau boost.
    double pVisRestFrame_parallel = pVisRestFrame*TMath::Cos(gjAngle);

    double enVisRestFrame = TMath::Sqrt(square(visMass) + square(pVisRestFrame));

    double gamma = (enVisRestFrame*TMath::Sqrt(square(enVisRestFrame) + square(pVisLabFrame_parallel) - square(pVisRestFrame_parallel)) 
                  - pVisRestFrame_parallel*pVisLabFrame_parallel)/(square(enVisRestFrame) - square(pVisRestFrame_parallel));

    double pMotherLabFrame = TMath::Sqrt(square(gamma) - 1)*motherMass;

    return pMotherLabFrame;
  }

  Vector motherDirection(const Vector& pVisLabFrame, double angleVisLabFrame, double phiLab) 
  {
    // The direction is defined using polar coordinates in a system where the visible energy
    // defines the Z axis.
    ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > motherDirectionVisibleSystem(1.0, angleVisLabFrame, phiLab);

    // Rotate into the LAB coordinate system
    return rotateUz(motherDirectionVisibleSystem, pVisLabFrame.Unit());
  }

  LorentzVector motherP4(const Vector& motherDirection, double motherMomentumLabFrame, double motherMass) 
  {
    // NB: tauDirection must be a unit vector !
    LorentzVector motherP4LabFrame(
                    motherDirection.x()*motherMomentumLabFrame,
		    motherDirection.y()*motherMomentumLabFrame,
		    motherDirection.z()*motherMomentumLabFrame,
		    TMath::Sqrt(motherMomentumLabFrame*motherMomentumLabFrame + motherMass*motherMass));
    return motherP4LabFrame;
  }

  void extractHistogramProperties(const TH1* histogram, const TH1* histogram_density,
                                  double& xMaximum, double& xMaximum_interpol, 
                                  double& xMean,
                                  double& xQuantile016, double& xQuantile050, double& xQuantile084,
                                  double& xMean3sigmaWithinMax, double& xMean5sigmaWithinMax, 
                                  int verbosity)
  {
    // compute median, -1 sigma and +1 sigma limits on reconstructed mass
    if ( verbosity ) std::cout << "<extractHistogramProperties>:" << std::endl;

    if ( histogram->Integral() > 0. ) {
      Double_t q[3];
      Double_t probSum[3];
      probSum[0] = 0.16;
      probSum[1] = 0.50;
      probSum[2] = 0.84;
      (const_cast<TH1*>(histogram))->GetQuantiles(3, q, probSum);
      xQuantile016 = q[0];
      xQuantile050 = q[1];
      xQuantile084 = q[2];
    } else {
      xQuantile016 = 0.;
      xQuantile050 = 0.;
      xQuantile084 = 0.;
    }
    
    xMean = histogram->GetMean();
    
    if ( histogram_density->Integral() > 0. ) {
      int binMaximum = histogram_density->GetMaximumBin();
      xMaximum = histogram_density->GetBinCenter(binMaximum);
      double yMaximum = histogram_density->GetBinContent(binMaximum);
      double yMaximumErr = ( histogram->GetBinContent(binMaximum) > 0. ) ?      
        (yMaximum*histogram->GetBinError(binMaximum)/histogram->GetBinContent(binMaximum)) : 0.;
      if ( verbosity ) std::cout << "yMaximum = " << yMaximum << " +/- " << yMaximumErr << " @ xMaximum = " << xMaximum << std::endl;
      if ( binMaximum > 1 && binMaximum < histogram_density->GetNbinsX() ) {
        int binLeft       = binMaximum - 1;
        double xLeft      = histogram_density->GetBinCenter(binLeft);
        double yLeft      = histogram_density->GetBinContent(binLeft);    
        
        int binRight      = binMaximum + 1;
        double xRight     = histogram_density->GetBinCenter(binRight);
        double yRight     = histogram_density->GetBinContent(binRight); 
        
        double xMinus     = xLeft - xMaximum;
        double yMinus     = yLeft - yMaximum;
        double xPlus      = xRight - xMaximum;
        double yPlus      = yRight - yMaximum;
        
        xMaximum_interpol = xMaximum + 0.5*(yPlus*square(xMinus) - yMinus*square(xPlus))/(yPlus*xMinus - yMinus*xPlus);
      } else {
        xMaximum_interpol = xMaximum;
      }
      if ( verbosity ) std::cout << "computing xMean3sigmaWithinMax:" << std::endl;
      xMean3sigmaWithinMax = getMeanOfBinsAboveThreshold(histogram_density, yMaximum - 3.*yMaximumErr, verbosity);
      if ( verbosity ) std::cout << "computing xMean5sigmaWithinMax:" << std::endl;
      xMean5sigmaWithinMax = getMeanOfBinsAboveThreshold(histogram_density, yMaximum - 5.*yMaximumErr, verbosity);
    } else {
      xMaximum = 0.;
      xMaximum_interpol = 0.;
      xMean3sigmaWithinMax = 0.;
      xMean5sigmaWithinMax = 0.;
    }
  }
}
