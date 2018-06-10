#include "copyright.h"
/*============================================================================*/
/*! \file init_grid.c 
 *  \brief Initializes most variables in the Grid structure.
 *
 * PURPOSE: Initializes most variables in the Grid structure.  Allocates memory
 *   for 3D arrays of Cons, interface B, etc.  With SMR, finds all overlaps
 *   between child and parent Grids, and initializes data needed for restriction
 *   flux-correction, and prolongation steps.
 *
 * CONTAINS PUBLIC FUNCTIONS: 
 * - init_grid()
 *
 * PRIVATE FUNCTION PROTOTYPES:
 * - checkOverlap() - checks for overlap of cubes, and returns overlap coords
 * - checkOverlapTouch() - same as above, but checks for overlap and/or touch */
/*============================================================================*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "defs.h"
#include "athena.h"
#include "globals.h"
#include "prototypes.h"

/*==============================================================================
 * PRIVATE FUNCTION PROTOTYPES:
 *  checkOverlap() - checks for overlap of cubes, and returns overlap coords
 *  checkOverlapTouch() - same as above, but checks for overlap and/or touch
 *============================================================================*/
#ifdef STATIC_MESH_REFINEMENT
/*! \fn int checkOverlap(SideS *pC1, SideS *pC2, SideS *pC3);
 *  \brief Checks for overlap of cubes, and returns overlap coords */
int checkOverlap(SideS *pC1, SideS *pC2, SideS *pC3);
/*! \fn int checkOverlapTouch(SideS *pC1, SideS *pC2, SideS *pC3);
 *  \brief Same as above, but checks for overlap and/or touch */
int checkOverlapTouch(SideS *pC1, SideS *pC2, SideS *pC3);
#endif

/*----------------------------------------------------------------------------*/
/*! \fn void init_grid(MeshS *pM)
 *  \brief Initializes most variables in the Grid structure.
 */

void init_grid(MeshS *pM)
{
  
  DomainS *pD;
  GridS *pG;
  int nDim,nl,nd,myL,myM,myN;
  int i,l,m,n,n1z,n2z,n3z;
#ifdef STATIC_MESH_REFINEMENT
  DomainS *pCD,*pPD;
  SideS D1,D2,D3,G1,G2,G3;
  int isDOverlap,isGOverlap,irefine,ncd,npd,dim,iGrid;
  int ncg,nCG,nMyCG,nCB[6],nMyCB[6],nb;
  int npg,nPG,nMyPG,nPB[6],nMyPB[6];
  int n1r,n2r,n1p,n2p;
#endif

  
/* number of dimensions in Grid. */
  nDim=1;
  for (i=1; i<3; i++) if (pM->Nx[i]>1) nDim++;

/* Loop over all levels and domains per level */

  for (nl=0; nl<pM->NLevels; nl++){
  for (nd=0; nd<pM->DomainsPerLevel[nl]; nd++){
    if (pM->Domain[nl][nd].Grid != NULL) {
      pD = (DomainS*)&(pM->Domain[nl][nd]);  /* set ptr to Domain */
      pG = pM->Domain[nl][nd].Grid;          /* set ptr to Grid */

      
   
      pG->time = pM->time;

/* get (l,m,n) coordinates of Grid being updated on this processor */

      get_myGridIndex(pD, myID_Comm_world, &myL, &myM, &myN);

/* ---------------------  Intialize grid in 1-direction --------------------- */
/* Initialize is,ie,dx1
 * Compute Disp, MinX[0], and MaxX[0] using displacement of Domain and Grid
 * location within Domain */

      pG->Nx[0] = pD->GData[myN][myM][myL].Nx[0];

      if(pG->Nx[0] > 1) {
        pG->is = nghost;
        pG->ie = pG->Nx[0] + nghost - 1;
 
      }
      else
        pG->is = pG->ie = 0;
    
      pG->dx1 = pD->dx[0];
    
      pG->Disp[0] = pD->Disp[0];
      pG->MinX[0] = pD->MinX[0];
      for (l=1; l<=myL; l++) {
        pG->Disp[0] +=        pD->GData[myN][myM][l-1].Nx[0];
        pG->MinX[0] += (Real)(pD->GData[myN][myM][l-1].Nx[0])*pG->dx1;
      }
      pG->MaxX[0] = pG->MinX[0] + (Real)(pG->Nx[0])*pG->dx1;
    
/* ---------------------  Intialize grid in 2-direction --------------------- */
/* Initialize js,je,dx2
 * Compute Disp, MinX[1], and MaxX[1] using displacement of Domain and Grid
 * location within Domain */

      pG->Nx[1] = pD->GData[myN][myM][myL].Nx[1];
    
      if(pG->Nx[1] > 1) {
        pG->js = nghost;
        pG->je = pG->Nx[1] + nghost - 1;

	/* printf("init_grid.c: %f %f \n", pG->js , pG->je ); */
	/* getchar(); */
      }
      else
        pG->js = pG->je = 0;
    
      pG->dx2 = pD->dx[1];

      pG->Disp[1] = pD->Disp[1];
      pG->MinX[1] = pD->MinX[1];
      for (m=1; m<=myM; m++) {
        pG->Disp[1] +=        pD->GData[myN][m-1][myL].Nx[1];
        pG->MinX[1] += (Real)(pD->GData[myN][m-1][myL].Nx[1])*pG->dx2;
      }
      pG->MaxX[1] = pG->MinX[1] + (Real)(pG->Nx[1])*pG->dx2;

/* ---------------------  Intialize grid in 3-direction --------------------- */
/* Initialize ks,ke,dx3
 * Compute Disp, MinX[2], and MaxX[2] using displacement of Domain and Grid
 * location within Domain */

      pG->Nx[2] = pD->GData[myN][myM][myL].Nx[2];

      if(pG->Nx[2] > 1) {
        pG->ks = nghost;
        pG->ke = pG->Nx[2] + nghost - 1;
      }
      else
        pG->ks = pG->ke = 0;

      pG->dx3 = pD->dx[2];

      pG->Disp[2] = pD->Disp[2];
      pG->MinX[2] = pD->MinX[2];
      for (n=1; n<=myN; n++) {
        pG->Disp[2] +=        pD->GData[n-1][myM][myL].Nx[2];
        pG->MinX[2] += (Real)(pD->GData[n-1][myM][myL].Nx[2])*pG->dx3;
      }
      pG->MaxX[2] = pG->MinX[2] + (Real)(pG->Nx[2])*pG->dx3;

/* ---------  Allocate 3D arrays to hold Cons based on size of grid --------- */

      if (pG->Nx[0] > 1)
        n1z = pG->Nx[0] + 2*nghost;
      else
        n1z = 1;

      if (pG->Nx[1] > 1)
        n2z = pG->Nx[1] + 2*nghost;
      else
        n2z = 1;

      if (pG->Nx[2] > 1)
        n3z = pG->Nx[2] + 2*nghost;
      else
        n3z = 1;

/* Build a 3D array of type ConsS */

      pG->U = (ConsS***)calloc_3d_array(n3z, n2z, n1z, sizeof(ConsS));
      if (pG->U == NULL) goto on_error1;
    
/* Build 3D arrays to hold interface field */

#ifdef MHD
      pG->B1i = (Real***)calloc_3d_array(n3z, n2z, n1z, sizeof(Real));
      if (pG->B1i == NULL) goto on_error2;

      pG->B2i = (Real***)calloc_3d_array(n3z, n2z, n1z, sizeof(Real));
      if (pG->B2i == NULL) goto on_error3;

      pG->B3i = (Real***)calloc_3d_array(n3z, n2z, n1z, sizeof(Real));
      if (pG->B3i == NULL) goto on_error4;
#endif /* MHD */

/* Build 3D arrays to magnetic diffusivities */

#ifdef RESISTIVITY
      pG->eta_Ohm = (Real***)calloc_3d_array(n3z, n2z, n1z, sizeof(Real));
      if (pG->eta_Ohm == NULL) goto on_error5;

      pG->eta_Hall = (Real***)calloc_3d_array(n3z, n2z, n1z, sizeof(Real));
      if (pG->eta_Hall == NULL) goto on_error6;

      pG->eta_AD = (Real***)calloc_3d_array(n3z, n2z, n1z, sizeof(Real));
      if (pG->eta_AD == NULL) goto on_error7;
#endif /* RESISTIVITY */
      // { printf("MPI breakpoint2, id= -1\n"); int mpi1= 1;  while(mpi1 == 1);}

  
/* Build 3D arrays related to Xray processing */
// anton
/* #ifdef XRAYS */
/*       //ionization parameter */
/*       pG->xi = (Real***)calloc_3d_array(n3z, n2z, n1z, sizeof(Real)); */
/*       if (pG->xi == NULL) goto on_error_xrays_xi; */

#ifdef XRAYS

    on_error_xrays_xi:
		free_3d_array(pG->xi);
    on_error_xrays_tau_e:
		free_3d_array(pG->tau_e);
    on_error_xrays_GridOfRays:
                free_3d_array(pG->GridOfRays);
		 ath_error("[init_grid]:on_error_xrays_GridOfRays\n");
		//a    /* on_error_xrays_disp: */
    /* 		free_3d_array(pG->disp); */

 on_error_xrays_yglob:

		;
		  
#ifdef use_glob_vars 
 free_3d_array(pG->yglob);
#endif 
		
#endif

      }

#ifdef STATIC_MESH_REFINEMENT
/*=========================== PRIVATE FUNCTIONS ==============================*/
/*----------------------------------------------------------------------------*/
/*! \fn int checkOverlap(SideS *pC1, SideS *pC2, SideS *pC3)
 *  \brief Checks if two cubes are overlapping.
 *
 *  - If yes returns true and sides of overlap region in Cube3
 *  - If no  returns false and -1 in Cube3
 *
 * Arguments are Side structures, containing indices of the 6 sides of cube */

int checkOverlap(SideS *pC1, SideS *pC2, SideS *pC3)
{
  int isOverlap=0;

  if (pC1->ijkl[0] < pC2->ijkr[0] && pC1->ijkr[0] > pC2->ijkl[0] &&
      pC1->ijkl[1] < pC2->ijkr[1] && pC1->ijkr[1] > pC2->ijkl[1] &&
      pC1->ijkl[2] < pC2->ijkr[2] && pC1->ijkr[2] > pC2->ijkl[2]) isOverlap=1;

  if (isOverlap==1) {
    pC3->ijkl[0] = MAX(pC1->ijkl[0], pC2->ijkl[0]);
    pC3->ijkr[0] = MIN(pC1->ijkr[0], pC2->ijkr[0]);
    pC3->ijkl[1] = MAX(pC1->ijkl[1], pC2->ijkl[1]);
    pC3->ijkr[1] = MIN(pC1->ijkr[1], pC2->ijkr[1]);
    pC3->ijkl[2] = MAX(pC1->ijkl[2], pC2->ijkl[2]);
    pC3->ijkr[2] = MIN(pC1->ijkr[2], pC2->ijkr[2]);
  } else {
    pC3->ijkl[0] = -1;
    pC3->ijkr[0] = -1;
    pC3->ijkl[1] = -1;
    pC3->ijkr[1] = -1;
    pC3->ijkl[2] = -1;
    pC3->ijkr[2] = -1;
  }

  return isOverlap;
}

/*----------------------------------------------------------------------------*/
/*! \fn int checkOverlapTouch(SideS *pC1, SideS *pC2, SideS *pC3)
 *  \brief Checks if two cubes are overlapping or touching.
 *
 *  - If yes returns true and sides of overlap region in Cube3
 *  - If no  returns false and -1 in Cube3
 *
 * Arguments are Side structures, containing indices of the 6 sides of cube */

int checkOverlapTouch(SideS *pC1, SideS *pC2, SideS *pC3)
{
  int isOverlap=0;

  if (pC1->ijkl[0] <= pC2->ijkr[0] && pC1->ijkr[0] >= pC2->ijkl[0] &&
      pC1->ijkl[1] <= pC2->ijkr[1] && pC1->ijkr[1] >= pC2->ijkl[1] &&
      pC1->ijkl[2] <= pC2->ijkr[2] && pC1->ijkr[2] >= pC2->ijkl[2]) isOverlap=1;

  if (isOverlap==1) {
    pC3->ijkl[0] = MAX(pC1->ijkl[0], pC2->ijkl[0]);
    pC3->ijkr[0] = MIN(pC1->ijkr[0], pC2->ijkr[0]);
    pC3->ijkl[1] = MAX(pC1->ijkl[1], pC2->ijkl[1]);
    pC3->ijkr[1] = MIN(pC1->ijkr[1], pC2->ijkr[1]);
    pC3->ijkl[2] = MAX(pC1->ijkl[2], pC2->ijkl[2]);
    pC3->ijkr[2] = MIN(pC1->ijkr[2], pC2->ijkr[2]);
  } else {
    pC3->ijkl[0] = -1;
    pC3->ijkr[0] = -1;
    pC3->ijkl[1] = -1;
    pC3->ijkr[1] = -1;
    pC3->ijkl[2] = -1;
    pC3->ijkr[2] = -1;
  }

  return isOverlap;
}
#endif /* STATIC_MESH_REFINEMENT */
