#include "TauAnalysis/SVfitStandalone/interface/SVfitStandaloneQuantities.h"

#include "Math/Factory.h"
#include "Math/Functor.h"
#include "Math/GSLMCIntegrator.h"
#include "Math/LorentzVector.h"
#include "Math/PtEtaPhiM4D.h"

#include <TGraphErrors.h>
#include <TMatrixD.h>
#include <TMatrixDSym.h>
#include <TMatrixDSymEigen.h>
#include <TVectorD.h>

namespace svFitStandalone
{
  TH1* makeHistogram(const std::string& histogramName, double xMin, double xMax, double logBinWidth)
  {
    if(xMin <= 0)
      xMin = 0.1;
    int numBins = 1 + TMath::Log(xMax/xMin)/TMath::Log(logBinWidth);
    TArrayF binning(numBins + 1);
    binning[0] = 0.;
    double x = xMin;  
    for ( int iBin = 1; iBin <= numBins; ++iBin ) {
      binning[iBin] = x;
      x *= logBinWidth;
    }  
    TH1* histogram = new TH1D(histogramName.data(), histogramName.data(), numBins, binning.GetArray());
    return histogram;
  }
  void compHistogramDensity(const TH1* histogram, TH1* histogram_density) 
  {
    for ( int iBin = 1; iBin <= histogram->GetNbinsX(); ++iBin ) {
      double binContent = histogram->GetBinContent(iBin);
      double binError = histogram->GetBinError(iBin);
      double binWidth = histogram->GetBinWidth(iBin);
      //if ( histogram == histogramMass_ ) {
      //  TAxis* xAxis = histogram->GetXaxis();
      //  std::cout << "bin #" << iBin << " (x = " << xAxis->GetBinLowEdge(iBin) << ".." << xAxis->GetBinUpEdge(iBin) << ", width = " << binWidth << "):"
      //	      << " " << binContent << " +/- " << binError << std::endl;
      //}
      assert(binWidth > 0.);
      histogram_density->SetBinContent(iBin, binContent/binWidth);
      histogram_density->SetBinError(iBin, binError/binWidth);
    }
  }
  double extractValue(const TH1* histogram, TH1* histogram_density) 
  {
    double maximum, maximum_interpol, mean, quantile016, quantile050, quantile084, mean3sigmaWithinMax, mean5sigmaWithinMax;
    compHistogramDensity(histogram, histogram_density);
    svFitStandalone::extractHistogramProperties(histogram, histogram_density, maximum, maximum_interpol, mean, quantile016, quantile050, quantile084, mean3sigmaWithinMax, mean5sigmaWithinMax);
    //double value = maximum_interpol;
    double value = maximum;
    return value;
  }
  double extractUncertainty(const TH1* histogram, TH1* histogram_density)
  {
    double maximum, maximum_interpol, mean, quantile016, quantile050, quantile084, mean3sigmaWithinMax, mean5sigmaWithinMax;
    compHistogramDensity(histogram, histogram_density);
    svFitStandalone::extractHistogramProperties(histogram, histogram_density, maximum, maximum_interpol, mean, quantile016, quantile050, quantile084, mean3sigmaWithinMax, mean5sigmaWithinMax);
    //double uncertainty = TMath::Sqrt(0.5*(TMath::Power(quantile084 - maximum_interpol, 2.) + TMath::Power(maximum_interpol - quantile016, 2.)));
    double uncertainty = TMath::Sqrt(0.5*(TMath::Power(quantile084 - maximum, 2.) + TMath::Power(maximum - quantile016, 2.)));
    return uncertainty;  
  }
  double extractLmax(const TH1* histogram, TH1* histogram_density)
  {
    compHistogramDensity(histogram, histogram_density);
    double Lmax = histogram_density->GetMaximum();
    return Lmax;
  }

  void map_xVEGAS(const double* x, bool l1isLep, bool l2isLep, bool marginalizeVisMass, bool shiftVisMass, bool shiftVisPt, double mvis, double mtest, double* x_mapped)
  {
    int offset1 = 0;
    if ( l1isLep ) {
      x_mapped[kXFrac]                 = x[0];
      x_mapped[kMNuNu]                 = x[1];
      x_mapped[kPhi]                   = x[2];
      x_mapped[kVisMassShifted]        = 0.;
      x_mapped[kRecTauPtDivGenTauPt]   = 0.;
      offset1 = 3;
    } else {
      x_mapped[kXFrac]                 = x[0];
      x_mapped[kMNuNu]                 = 0.;
      x_mapped[kPhi]                   = x[1];
      offset1 = 2;
      if ( marginalizeVisMass || shiftVisMass ) {
	x_mapped[kVisMassShifted]      = x[offset1];
	++offset1;
      } else {
	x_mapped[kVisMassShifted]      = 0.;
      }
      if ( shiftVisPt ) {
	x_mapped[kRecTauPtDivGenTauPt] = x[offset1];
	++offset1;
      } else {
	x_mapped[kRecTauPtDivGenTauPt] = 0.;
      }
    }    
    double x2 = ( x[0] > 0. ) ? TMath::Power(mvis/mtest, 2.)/x[0] : 1.e+3;
    int offset2 = 0;
    if ( l2isLep ) {
      x_mapped[kMaxFitParams + kXFrac]                 = x2;
      x_mapped[kMaxFitParams + kMNuNu]                 = x[offset1 + 0];
      x_mapped[kMaxFitParams + kPhi]                   = x[offset1 + 1];
      x_mapped[kMaxFitParams + kVisMassShifted]        = 0.;
      x_mapped[kMaxFitParams + kRecTauPtDivGenTauPt]   = 0.;
      offset2 = 2;
    } else {
      x_mapped[kMaxFitParams + kXFrac]                 = x2;
      x_mapped[kMaxFitParams + kMNuNu]                 = 0.;
      x_mapped[kMaxFitParams + kPhi]                   = x[offset1 + 0];
      offset2 = 1;
      if ( marginalizeVisMass || shiftVisMass ) {
	x_mapped[kMaxFitParams + kVisMassShifted]      = x[offset1 + offset2];
	++offset2;
      } else {
	x_mapped[kMaxFitParams + kVisMassShifted]      = 0.;
      }
      if ( shiftVisPt ) {
	x_mapped[kMaxFitParams + kRecTauPtDivGenTauPt] = x[offset1 + offset2];
	++offset2;
      } else {
	x_mapped[kMaxFitParams + kRecTauPtDivGenTauPt] = 0.;
      }
    }
  }

  void map_xMarkovChain(const double* x, bool l1isLep, bool l2isLep, bool marginalizeVisMass, bool shiftVisMass, bool shiftVisPt, double* x_mapped)
  {
    int offset1 = 0;
    if ( l1isLep ) {
      x_mapped[kXFrac]                 = x[0];
      x_mapped[kMNuNu]                 = x[1];
      x_mapped[kPhi]                   = x[2];
      x_mapped[kVisMassShifted]        = 0.;
      x_mapped[kRecTauPtDivGenTauPt]   = 0.;
      offset1 = 3;
    } else {
      x_mapped[kXFrac]                 = x[0];
      x_mapped[kMNuNu]                 = 0.;
      x_mapped[kPhi]                   = x[1];
      offset1 = 2;
      if ( marginalizeVisMass || shiftVisMass ) {
	x_mapped[kVisMassShifted]      = x[offset1];
	++offset1;
      } else {
	x_mapped[kVisMassShifted]      = 0.;
      }
      if ( shiftVisPt ) {
	x_mapped[kRecTauPtDivGenTauPt] = x[offset1];
	++offset1;
      } else {
	x_mapped[kRecTauPtDivGenTauPt] = 0.;
      }
    }
    int offset2 = 0;
    if ( l2isLep ) {
      x_mapped[kMaxFitParams + kXFrac]                 = x[offset1 + 0];
      x_mapped[kMaxFitParams + kMNuNu]                 = x[offset1 + 1];
      x_mapped[kMaxFitParams + kPhi]                   = x[offset1 + 2];
      x_mapped[kMaxFitParams + kVisMassShifted]        = 0.;
      x_mapped[kMaxFitParams + kRecTauPtDivGenTauPt]   = 0.;
      offset2 = 3;
    } else {
      x_mapped[kMaxFitParams + kXFrac]                 = x[offset1 + 0];
      x_mapped[kMaxFitParams + kMNuNu]                 = 0.;
      x_mapped[kMaxFitParams + kPhi]                   = x[offset1 + 1];
      offset2 = 2;
      if ( marginalizeVisMass || shiftVisMass ) {
	x_mapped[kMaxFitParams + kVisMassShifted]      = x[offset1 + offset2];
	++offset2;
      } else {
	x_mapped[kMaxFitParams + kVisMassShifted]      = 0.;
      }
      if ( shiftVisPt ) {	
	x_mapped[kMaxFitParams + kRecTauPtDivGenTauPt] = x[offset1 + offset2];
	++offset2;
      } else {
	x_mapped[kMaxFitParams + kRecTauPtDivGenTauPt] = 0.;
      }
    }
    //std::cout << "<map_xMarkovChain>:" << std::endl;
    //for ( int i = 0; i < 6; ++i ) {
    //  std::cout << " x_mapped[" << i << "] = " << x_mapped[i] << std::endl;
    //}
  }
  
  SVfitQuantity::SVfitQuantity(
      TH1* histogram,
      TH1* histogram_density,
      std::function<double(std::vector<svFitStandalone::LorentzVector> const&, std::vector<MeasuredTauLepton> const&, double, double, TMatrixD const&) > function
  ) :
    histogram_(histogram),
    histogram_density_(histogram_density),
    function_(function)
  {
  }
  SVfitQuantity::~SVfitQuantity()
  {
    if (histogram_ != nullptr) delete histogram_;
    if (histogram_density_ != nullptr) delete histogram_density_;
  }
  void SVfitQuantity::SetHistograms(TH1* histogram, TH1* histogram_density)
  {
    // CV: passing null pointers to the SetHistogramMass function
    //     indicates that the histograms have been deleted by the calling code
    if (histogram_ != nullptr) delete histogram_;
    histogram_ = histogram;
    if (histogram_density_ != nullptr) delete histogram_density_;
    histogram_density_ = histogram_density;
  }
  void SVfitQuantity::Reset()
  {
    histogram_->Reset();
    histogram_density_->Reset();
  }
  void SVfitQuantity::WriteHistograms() const
  {
    if (histogram_ != nullptr) histogram_->Write();
    if (histogram_density_ != nullptr) histogram_density_->Write();
  }
  double SVfitQuantity::Eval(
      std::vector<svFitStandalone::LorentzVector> const& fittedTauLeptons,
      std::vector<MeasuredTauLepton> const& measuredTauLeptons,
      double measuredMETx,
      double measuredMETy,
      TMatrixD const& covME
  ) const
  {
    return function_(fittedTauLeptons, measuredTauLeptons, measuredMETx, measuredMETy, covME);
  }
  
  double SVfitQuantity::ExtractValue() const
  {
    return extractValue(histogram_, histogram_density_);
  }
  double SVfitQuantity::ExtractUncertainty() const
  {
    return extractUncertainty(histogram_, histogram_density_);
  }
  double SVfitQuantity::ExtractLmax() const
  {
    return extractLmax(histogram_, histogram_density_);
  }
  
  MCQuantitiesAdapter::MCQuantitiesAdapter(std::vector<SVfitQuantity*> const& quantities) :
    quantities_(quantities)
  {
    quantities_.push_back(new SVfitQuantity(makeHistogram("SVfitStandaloneAlgorithm_histogramMass", 1.e+1, 1.e+4, 1.025),
                                            makeHistogram("SVfitStandaloneAlgorithm_histogramMass_density", 1.e+1, 1.e+4, 1.025),
                                            [](std::vector<svFitStandalone::LorentzVector> const& fittedTauLeptons, std::vector<MeasuredTauLepton> const&, double, double, TMatrixD const&) -> double
                                            {
                                              return (fittedTauLeptons.at(0) + fittedTauLeptons.at(1)).mass();
                                            }));
    quantities_.push_back(new SVfitQuantity(makeHistogram("SVfitStandaloneAlgorithm_histogramTransverseMass", 1., 1.e+4, 1.025),
                                            makeHistogram("SVfitStandaloneAlgorithm_histogramTransverseMass_density", 1., 1.e+4, 1.025),
                                            [](std::vector<svFitStandalone::LorentzVector> const& fittedTauLeptons, std::vector<MeasuredTauLepton> const&, double, double, TMatrixD const&) -> double
                                            {
                                              return TMath::Sqrt(2.0*fittedTauLeptons.at(0).pt()*fittedTauLeptons.at(1).pt()*(1.0 - TMath::Cos(fittedTauLeptons.at(0).phi() - fittedTauLeptons.at(1).phi())));
                                            }));
  }
  MCQuantitiesAdapter::~MCQuantitiesAdapter()
  {
    for (std::vector<SVfitQuantity*>::iterator quantity = quantities_.begin(); quantity != quantities_.end(); ++quantity)
    {
      delete *quantity;
    }
  }
  void MCQuantitiesAdapter::SetHistograms(size_t index, TH1* histogram, TH1* histogram_density)
  {
    quantities_.at(index)->SetHistograms(histogram, histogram_density);
  }
  void MCQuantitiesAdapter::SetHistograms(std::vector<TH1*> histograms, std::vector<TH1*> histogram_densities)
  {
    for (size_t index = 0; index != quantities_.size(); ++index)
    {
      quantities_.at(index)->SetHistograms(histograms.at(index), histogram_densities.at(index));
    }
  }
  void MCQuantitiesAdapter::SetHistogramMass(TH1* histogram, TH1* histogram_density)
  {
    SetHistograms(GetNQuantities()-2, histogram, histogram_density);
  }
  void MCQuantitiesAdapter::SetHistogramTransverseMass(TH1* histogram, TH1* histogram_density)
  {
    SetHistograms(GetNQuantities()-1, histogram, histogram_density);
  }
  void MCQuantitiesAdapter::Reset()
  {
    for (std::vector<SVfitQuantity*>::iterator quantity = quantities_.begin(); quantity != quantities_.end(); ++quantity)
    {
      (*quantity)->Reset();
    }
  }
  void MCQuantitiesAdapter::WriteHistograms() const
  {
    for (std::vector<SVfitQuantity*>::const_iterator quantity = quantities_.begin(); quantity != quantities_.end(); ++quantity)
    {
      (*quantity)->WriteHistograms();
    }
  }
  double MCQuantitiesAdapter::DoEval(const double* x) const
  {
    map_xMarkovChain(x, l1isLep_, l2isLep_, marginalizeVisMass_, shiftVisMass_, shiftVisPt_, x_mapped_);
    SVfitStandaloneLikelihood::gSVfitStandaloneLikelihood->results(fittedTauLeptons_, x_mapped_);
    for (std::vector<SVfitQuantity*>::const_iterator quantity = quantities_.begin(); quantity != quantities_.end(); ++quantity)
    {
      //(*quantity)->histogram_->Fill((*quantity)->Eval(fittedTauLeptons_, measuredTauLeptons_rounded, measuredMETx_rounded, measuredMETy_rounded, covMET_rounded));
      (*quantity)->histogram_->Fill((*quantity)->Eval(fittedTauLeptons_, std::vector<MeasuredTauLepton>(), 0.0, 0.0, TMatrixD(2,2)));
    }
    return 0.0;
  }
  double MCQuantitiesAdapter::ExtractValue(size_t index) const
  {
    return quantities_.at(index)->ExtractValue();
  }
  double MCQuantitiesAdapter::ExtractUncertainty(size_t index) const
  {
    return quantities_.at(index)->ExtractUncertainty();
  }
  double MCQuantitiesAdapter::ExtractLmax(size_t index) const
  {
    return quantities_.at(index)->ExtractLmax();
  }
  std::vector<double> MCQuantitiesAdapter::ExtractValues() const
  {
  	std::vector<double> results;
  	std::transform(quantities_.begin(), quantities_.end(), results.begin(),
                   [](SVfitQuantity* quantity) { return quantity->ExtractValue(); });
    return results;
  }
  std::vector<double> MCQuantitiesAdapter::ExtractUncertainties() const
  {
  	std::vector<double> results;
  	std::transform(quantities_.begin(), quantities_.end(), results.begin(),
                   [](SVfitQuantity* quantity) { return quantity->ExtractUncertainty(); });
    return results;
  }
  std::vector<double> MCQuantitiesAdapter::ExtractLmaxima() const
  {
  	std::vector<double> results;
  	std::transform(quantities_.begin(), quantities_.end(), results.begin(),
                   [](SVfitQuantity* quantity) { return quantity->ExtractLmax(); });
    return results;
  }
  double MCQuantitiesAdapter::getMass() const { return ExtractValue(quantities_.size()-2); }
  double MCQuantitiesAdapter::getMassUncert() const { return ExtractUncertainty(quantities_.size()-2); }
  double MCQuantitiesAdapter::getMassLmax() const { return ExtractLmax(quantities_.size()-2); }
  double MCQuantitiesAdapter::getTransverseMass() const { return ExtractValue(quantities_.size()-1); }
  double MCQuantitiesAdapter::getTransverseMassUncert() const { return ExtractUncertainty(quantities_.size()-1); }
  double MCQuantitiesAdapter::getTransverseMassLmax() const { return ExtractLmax(quantities_.size()-1); }

  MCPtEtaPhiMassAdapter::MCPtEtaPhiMassAdapter() :
    MCQuantitiesAdapter({
      new SVfitQuantity(makeHistogram("SVfitStandaloneAlgorithm_histogramPt", 1., 1.e+3, 1.025),
                        makeHistogram("SVfitStandaloneAlgorithm_histogramPt_density", 1., 1.e+3, 1.025),
                        [](std::vector<svFitStandalone::LorentzVector> const& fittedTauLeptons, std::vector<MeasuredTauLepton> const&, double, double, TMatrixD const&) -> double
                        {
                          return (fittedTauLeptons.at(0) + fittedTauLeptons.at(1)).pt();
                        }),
      new SVfitQuantity(new TH1D("SVfitStandaloneAlgorithm_histogramEta", "SVfitStandaloneAlgorithm_histogramEta", 198, -9.9, +9.9),
                        new TH1D("SVfitStandaloneAlgorithm_histogramEta_density", "SVfitStandaloneAlgorithm_histogramEta_density", 198, -9.9, +9.9),
                        [](std::vector<svFitStandalone::LorentzVector> const& fittedTauLeptons, std::vector<MeasuredTauLepton> const&, double, double, TMatrixD const&) -> double
                        {
                          return (fittedTauLeptons.at(0) + fittedTauLeptons.at(1)).eta();
                        }),
      new SVfitQuantity(new TH1D("SVfitStandaloneAlgorithm_histogramPhi", "SVfitStandaloneAlgorithm_histogramPhi", 180, -TMath::Pi(), +TMath::Pi()),
                        new TH1D("SVfitStandaloneAlgorithm_histogramPhi_density", "SVfitStandaloneAlgorithm_histogramPhi_density", 180, -TMath::Pi(), +TMath::Pi()),
                        [](std::vector<svFitStandalone::LorentzVector> const& fittedTauLeptons, std::vector<MeasuredTauLepton> const&, double, double, TMatrixD const&) -> double
                        {
                          return (fittedTauLeptons.at(0) + fittedTauLeptons.at(1)).phi();
                        })
    })
  {
  }
  double MCPtEtaPhiMassAdapter::getPt() const { return ExtractValue(0); }
  double MCPtEtaPhiMassAdapter::getPtUncert() const { return ExtractUncertainty(0); }
  double MCPtEtaPhiMassAdapter::getPtLmax() const { return ExtractLmax(0); }
  double MCPtEtaPhiMassAdapter::getEta() const { return ExtractValue(1); }
  double MCPtEtaPhiMassAdapter::getEtaUncert() const { return ExtractUncertainty(1); }
  double MCPtEtaPhiMassAdapter::getEtaLmax() const { return ExtractLmax(1); }
  double MCPtEtaPhiMassAdapter::getPhi() const { return ExtractValue(2); }
  double MCPtEtaPhiMassAdapter::getPhiUncert() const { return ExtractUncertainty(2); }
  double MCPtEtaPhiMassAdapter::getPhiLmax() const { return ExtractLmax(2); }
}

