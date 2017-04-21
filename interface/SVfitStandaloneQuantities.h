#ifndef TauAnalysis_SVfitStandalone_SVfitStandaloneQuantities_h
#define TauAnalysis_SVfitStandalone_SVfitStandaloneQuantities_h

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
  TH1* compHistogramDensity(const TH1*);
  double extractValue(const TH1*);
  double extractUncertainty(const TH1*);
  double extractLmax(const TH1*);

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
    SVfitQuantity();
    virtual ~SVfitQuantity();

    virtual TH1* CreateHistogram(std::vector<svFitStandalone::LorentzVector> const& measuredTauLeptons, svFitStandalone::Vector const& measuredMET) const = 0;
    virtual double FitFunction(std::vector<svFitStandalone::LorentzVector> const& fittedTauLeptons, std::vector<svFitStandalone::LorentzVector> const& measuredTauLeptons, svFitStandalone::Vector const& measuredMET) const = 0;

    void SetHistogram(std::vector<svFitStandalone::LorentzVector> const& measuredTauLeptons, svFitStandalone::Vector const& measuredMET);
    void Reset();
    void WriteHistograms() const;

    double Eval(
        std::vector<svFitStandalone::LorentzVector> const& fittedTauLeptons,
        std::vector<svFitStandalone::LorentzVector> const& measuredTauLeptons,
        svFitStandalone::Vector const& measuredMET
    ) const;

    double ExtractValue() const;
    double ExtractUncertainty() const;
    double ExtractLmax() const;

    mutable TH1* histogram_ = nullptr;
  };

  class HiggsPtSVfitQuantity : public SVfitQuantity
  {
   public:
    virtual TH1* CreateHistogram(std::vector<svFitStandalone::LorentzVector> const& measuredTauLeptons, svFitStandalone::Vector const& measuredMET) const;
    virtual double FitFunction(std::vector<svFitStandalone::LorentzVector> const& fittedTauLeptons, std::vector<svFitStandalone::LorentzVector> const& measuredTauLeptons, svFitStandalone::Vector const& measuredMET) const;
  };
  class HiggsEtaSVfitQuantity : public SVfitQuantity
  {
   public:
    virtual TH1* CreateHistogram(std::vector<svFitStandalone::LorentzVector> const& measuredTauLeptons, svFitStandalone::Vector const& measuredMET) const;
    virtual double FitFunction(std::vector<svFitStandalone::LorentzVector> const& fittedTauLeptons, std::vector<svFitStandalone::LorentzVector> const& measuredTauLeptons, svFitStandalone::Vector const& measuredMET) const;
  };
  class HiggsPhiSVfitQuantity : public SVfitQuantity
  {
   public:
    virtual TH1* CreateHistogram(std::vector<svFitStandalone::LorentzVector> const& measuredTauLeptons, svFitStandalone::Vector const& measuredMET) const;
    virtual double FitFunction(std::vector<svFitStandalone::LorentzVector> const& fittedTauLeptons, std::vector<svFitStandalone::LorentzVector> const& measuredTauLeptons, svFitStandalone::Vector const& measuredMET) const;
  };
  class HiggsMassSVfitQuantity : public SVfitQuantity
  {
   public:
    virtual TH1* CreateHistogram(std::vector<svFitStandalone::LorentzVector> const& measuredTauLeptons, svFitStandalone::Vector const& measuredMET) const;
    virtual double FitFunction(std::vector<svFitStandalone::LorentzVector> const& fittedTauLeptons, std::vector<svFitStandalone::LorentzVector> const& measuredTauLeptons, svFitStandalone::Vector const& measuredMET) const;
  };
  class TransverseMassSVfitQuantity : public SVfitQuantity
  {
   public:
    virtual TH1* CreateHistogram(std::vector<svFitStandalone::LorentzVector> const& measuredTauLeptons, svFitStandalone::Vector const& measuredMET) const;
    virtual double FitFunction(std::vector<svFitStandalone::LorentzVector> const& fittedTauLeptons, std::vector<svFitStandalone::LorentzVector> const& measuredTauLeptons, svFitStandalone::Vector const& measuredMET) const;
  };

  class MCQuantitiesAdapter : public ROOT::Math::Functor
  {
   public:
    MCQuantitiesAdapter(std::vector<SVfitQuantity*> const& quantities = std::vector<SVfitQuantity*>());
    ~MCQuantitiesAdapter();

    void SetMeasurements(std::vector<svFitStandalone::LorentzVector> const& measuredTauLeptons, svFitStandalone::Vector const& measuredMET);
    void SetHistograms(std::vector<svFitStandalone::LorentzVector> const& measuredTauLeptons, svFitStandalone::Vector const& measuredMET);
    void Reset();
    void WriteHistograms() const;

    inline void SetL1isLep(bool l1isLep) { l1isLep_ = l1isLep; }
    inline void SetL2isLep(bool l2isLep) { l2isLep_ = l2isLep; }
    inline void SetMarginalizeVisMass(bool marginalizeVisMass) { marginalizeVisMass_ = marginalizeVisMass; }
    inline void SetShiftVisMass(bool shiftVisMass) { shiftVisMass_ = shiftVisMass; }
    inline void SetShiftVisPt(bool shiftVisPt) { shiftVisPt_ = shiftVisPt; }
    void SetNDim(unsigned int nDim) { nDim_ = nDim; }

    unsigned int NDim() const { return nDim_; }

    unsigned int GetNQuantities() const { return quantities_.size(); }
    double ExtractValue(size_t index) const;
    double ExtractUncertainty(size_t index) const;
    double ExtractLmax(size_t index) const;

    std::vector<double> ExtractValues() const;
    std::vector<double> ExtractUncertainties() const;
    std::vector<double> ExtractLmaxima() const;

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

    std::vector<svFitStandalone::LorentzVector> measuredTauLeptons_;
    svFitStandalone::Vector measuredMET_;

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
    double getMass() const;
    double getMassUncert() const;
    double getMassLmax() const;
    double getTransverseMass() const;
    double getTransverseMassUncert() const;
    double getTransverseMassLmax() const;
  };
}

#endif
