#ifndef TauAnalysis_SVfitStandalone_LikelihoodFunctions_h
#define TauAnalysis_SVfitStandalone_LikelihoodFunctions_h

#include "TMatrixD.h"

/**
   \class   probMET LikelihoodFunctions.h "TauAnalysis/CandidateTools/interface/LikelihoodFunctions.h"
   
   \brief   Likelihood for MET

   Likelihood for MET. Input parameters are:

    pMETX  : difference between reconstructed MET and fitted MET in x
    pMETY  : difference between reconstructed MET and fitted MET in y
    covDet : determinant of the covariance matrix of the measured MET
    covInv : 2-dim inverted covariance matrix for measured MET (to be 
             determined from MET significance algorithm)
    power  : additional power to enhance the nll term
*/
double probMET(double dMETX, double dMETY, double covDet, const TMatrixD& covInv, double power = 1., bool verbose = false);

/**
   \class   probTauToLepPhaseSpace LikelihoodFunctions.h "TauAnalysis/CandidateTools/interface/LikelihoodFunctions.h"
   
   \brief   Matrix Element for leptonic tau decays. Input parameters are:

   \var decayAngle : decay angle in the restframe of the tau lepton decay
   \var visMass : measured visible mass
   \var nunuMass : fitted neutrino mass

   The parameter tauLeptonMass2 is the mass of the tau lepton squared as defined in svFitStandaloneAuxFunctions.h    

     NOTE: The formulas taken from the paper
             "Tau polarization and its correlations as a probe of new physics",
             B.K. Bullock, K. Hagiwara and A.D. Martin,
             Nucl. Phys. B395 (1993) 499.

*/
double probTauToLepMatrixElement(double decayAngle, double nunuMass, double visMass, double x, bool applySinTheta, bool verbose = false);

/**
   \class   probTauToHadPhaseSpace LikelihoodFunctions.h "TauAnalysis/CandidateTools/interface/LikelihoodFunctions.h"
   
   \brief   Likelihood for a two body tau decay into two hadrons

   Likelihood for a two body tau decay into two hadrons. Input parameters is:

    decayAngle : decay angle in the restframe of the tau lepton decay
*/
double probTauToHadPhaseSpace(double decayAngle, double nunuMass, double visMass, double x, bool applySinTheta, bool verbose = false);

#endif
