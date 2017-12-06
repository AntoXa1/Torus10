#include "copyright.h"
/*============================================================================*/
/*! \file hkdisk.c
 *  \brief Problem generator for Hawley Krolik disk (Specifically, GT4)
 */
/*============================================================================*/

#include <math.h>
#include <stdio.h>

#include <unistd.h>

#include <stdlib.h>
#include <string.h>
#include "defs.h"
#include "athena.h"
#include "globals.h"
#include "prototypes.h"

//****************************************

#ifdef MPI_PARALLEL

//#define DBG_MPI_OPT_STACK //mpi "breakpoint" while loops

#endif

#define HAWL

//#define RADIAL_RAY_TEST
//#define SOLOV



//****************************************
/* make all MACHINE=macosxmpi */


static void inX1BoundCond(GridS *pGrid);
static void diode_outflow_ix3(GridS *pGrid);
static void diode_outflow_ox3(GridS *pGrid);

void plot(MeshS *pM, char name[16]);
void aplot(MeshS *pM, int is, int js, int ks, int ie, int je, int ke, char name[16]);


/* void aplot(GridS *pG, int is, int ie, int js, int je, int ks, int ke); */
//void plot(MeshS *pM, char name[16]);

static void calcProblemParameters();
static void printProblemParameters();

#ifdef XRAYS
//void optDepthFunctions(MeshS *pM);

void Constr_optDepthStack(MeshS *pM, GridS *pG);

void Constr_optDepthStackOnGlobGrid(MeshS *pM, GridS *pG);

void optDepthStack(MeshS *pM, GridS *pG);

void optDepthStackOnGlobGrid(MeshS *pM, GridS *pG, int my_id);
void ionizParam(const MeshS *pM, GridS *pG);
void optDepthFunctions(GridS *pG);


Real updateEnergyFromXrayHeatCool(const Real E, const Real d,
				  const Real M1,const Real M2, const Real M3,
				  const Real B1c,const Real B2c,const Real B3c,
				  const Real xi, const Real dt,
				  int i, int j, int k);



void xRayHeatCool(const Real dens, const Real Press, const Real xi_in, Real*, Real*,  const Real dt);


Real rtsafe_energy_eq(Real, Real, Real, Real, int*);
#endif

// Input parameters
static Real q, r0, r_in, rho0, e0, dcut, beta, seed, R_ib;
static Real nc0, F2Fedd, fx =0.5, fuv =0.5;
static Real rRadSrc[2]={0.,0.}; //coordinates of the radiation source
static Real Ctor, Kbar, nAd, q1, xm, rgToR0,Tx,Rg,MBH_in_gram,Tgmin,
  Dsc,Nsc,Rsc,Usc,Psc, Esc,Time0,Tvir0,Lamd0,Evir0, Ledd_e, inRadOfMesh,
  Lx, Luv, a1;


static double
  CL = 2.997925E10,
  GRV = 6.67384E-8,
  QE=4.80325E-10,
  MSUN = 1.989E33,
  M2Msun = 10E7,
  GAM53 = 5./3.,
  KPE = 0.4,
  PC = 3.085678E18,
  ARAD = 7.56464E-15,
  RGAS = 8.31E7,
  MP = 1.672661E-24,
  M_U = 1.660531E-24,
  RSUN = 6.95E10,
  SGMB,MSOLYR,
  YR = 365.*24.*3600.,
  M_MW = 1., //mean mol. weight
  tiny = 1.E-10;
//PI = 3.14159265359;
//----------------------------


#ifdef SOLOV
static Real
  Kbar_Sol,
  Br0_Sol = 1.,
  Br1_Sol = 6.,

  a1_Sol = 1,
  a2_Sol = 1,
  b1_Sol = 3,

  w0_Sol = 2.,
  w1_Sol = 0.;
#endif
/* ============================================================================= */
/*                       ray tracing functions                                   */
/* ============================================================================= */
void testRayTracings( MeshS *pM, GridS *pG);
void traceGridCell(GridS *pG, Real *res, int ijk_cur[3], Real xpos[3], Real* x1x2x3, const Real
		   cartnorm[3], const Real cylnorm[3], short*);
void traceGridCellOnGlobGrid(MeshS *pM,GridS *pG, Real *res, int *ijk_cur, Real *xyz_pos,
		   Real* rtz_pos, const Real *cart_norm, const Real *cyl_norm,			    
			     short* nroot);
void coordCylToCart(Real*, const Real*,const Real, const Real );
void cartVectorToCylVector(Real*, const Real*, const Real, const Real);
void vectBToA(Real*, const Real*, const Real*);
void normVectBToA(Real* v, const Real* , const Real*, Real* );
Real absValueVector(const Real*);

void cc_posGlobFromGlobIndx(const MeshS *pM, const GridS *pG, const int iglob, const int jglob,const int kglob, Real *px1, Real *px2, Real *px3);

void cc_posGlob(const MeshS *pM, const GridS *pG, const int iloc, const int jloc,const int kloc,
		Real *px1, Real *px2, Real *px3);
void ijkLocToGlob(const GridS *pG, const int iloc, const int jloc,const int kloc,
		  int *iglob, int *jglob, int *kglob);

void ijkGlobToLoc(const GridS *pG, const int iglob, const int jglob, const int kglob, int *iloc, int *jloc, int *kloc);


int celli_Glob(const MeshS *pM, const Real x, const Real dx1_1, int *i, Real *a, const Real is);
int cellj_Glob(const MeshS *pM, const Real y, const Real dx2_1, int *j, Real *b,const Real js);
int cellk_Glob(const MeshS *pM, const Real z, const Real dx3_1, int *k, Real *c, const Real ks);

#ifdef MPI_PARALLEL

void initGlobComBuffer(const MeshS *pM, const GridS *pG );

void SyncGridGlob(MeshS *pM, DomainS *pDomain, GridS *pG, int W2Do);

int ID_DEN = 0;
int ID_TAU = 1;

static void packGridForGlob(MeshS *pM, GridS *pG, int* my_id, int W2Do);

static void packGlobBufForGlobSync(MeshS *pM, GridS *pG, int* myid, int W2Do);

static void unPackAndFetchToGlobGrid(MeshS *pM, GridS *pG, int* ext_id, int W2Do);
static void unPackGlobBufForGlobSync(MeshS *pM, GridS *pG, int* ext_id, int W2Do);

void freeGlobArrays();


/* MPI send and receive buffers */
int  BufSize, BufSizeGlobArr, ibufe, jbufe, kbufe, **BufBndr, NumDomsInGlob; //sizes of global buffer 
static double **send_buf = NULL, **recv_buf = NULL,
  **send_buf_big = NULL, **recv_buf_big = NULL;
static MPI_Request *recv_rq, *send_rq;
#endif /* MPI_PARALLEL */
/* ============================================================================= */
/*                      boundary conditions                                      */
/* ============================================================================= */


//extern x1GravAcc;

/*! \fn static Real grav_pot(const Real x1, const Real x2, const Real x3)
 *  \brief Gravitational potential */


void getIndexOfCellByZ(GridS *pG, Real z, int  *k){
  *k= (int)( (z - pG->MinX[2])/pG->dx3 -0.5 ) + pG->ks;
  /* printf(" %d, \n",*k); getchar(); */
  }
//	*px1 = pG->MinX[0] + ((Real)(i - pG->is) + 0.5)*pG->dx1;
//	*px2 = pG->MinX[1] + ((Real)(j - pG->js) + 0.5)*pG->dx2;
//	*px3 = pG->MinX[2] + ((Real)(k - pG->ks) + 0.5)*pG->dx3;


void apause(){
  printf("pause.. press any key ss... ");
  getchar();
}


#ifdef XRAYS

void optDepthStack( MeshS *pM, GridS *pG){

  int i,j,k,is,ie,js,je,ks,ke,ip,jp,kp,i_g, j_g, k_g,ii;
  Real den,dtau,dl,tau;
 
  is = pG->is;
  ie = pG->ie;
  js = pG->js;
  je = pG->je;
  ks = pG->ks;
  ke = pG->ke;


  for (kp = ks; kp<=ke; kp++) { //z
    for (jp = js; jp<=je; jp++) { //t
      for (ip = is; ip<=ie; ip++) { //r
	tau=0;    	

	/* printf(" %d %d \n", ip, is); */
	       
	/* if (ip == 0){ /\* first index on Mesh *\/ */
	/*   tau = pG->tau_e[kp][jp][ip-1]; */
	/* } */
	
	for (ii = 0; ii<(pG->GridOfRays[kp][jp][ip]).len; ii++){	  
	  
	  i = (pG->GridOfRays[kp][jp][ip]).Ray[ii].i;
	  j = (pG->GridOfRays[kp][jp][ip]).Ray[ii].j;
	  k = (pG->GridOfRays[kp][jp][ip]).Ray[ii].k;

	  ijkLocToGlob(pG, i,j,k, &i_g, &j_g, &k_g);

	  den = pG->yglob[k_g][j_g][i_g].ro;

	  dl = (pG->GridOfRays[kp][jp][ip].Ray[ii]).dl;
	
	  dtau = Rsc*Dsc*KPE* dl*den;		
	  
	  tau+=dtau;
			
	  /* printf(" %d \n", (pG->GridOfRays[kp][jp][ip]).len); */
			
	}
	pG->tau_e[kp][jp][ip] = tau;

      } //ip
    } //jp
  } //kp

}
void optDepthStackOnGlobGrid(MeshS *pM, GridS *pG, int my_id){
 /* by the time this function is executed, density must be synced on global grid */
  
  int i,j,k,is,ie,js,je,ks,ke, il, iu, jl,ju,kl,ku,ip,jp,kp, iloc,jloc,kloc,
 	 ii;
  Real den,dtau,dl,tau;
   
  is = 0;
  ie = pM->Nx[0]-1;     
  js = 0;
  je = pM->Nx[1]-1;       
  ks = 0;
  ke = pM->Nx[2]-1;
 

   /* printf(" in optDepthStackOnGlobGrid:  %d  %d  %d %d  %d  %d  id = %d \n\n", */
   /* 	  BufBndr[my_id][0], BufBndr[my_id][1], BufBndr[my_id][2], */
   /* 	  BufBndr[my_id][3], BufBndr[my_id][4], BufBndr[my_id][5], */
   /* 	  my_id); */

   /* printf("is, ie = %d %d id = %d \n", BufBndr[my_id][0], BufBndr[my_id][3], my_id); */
   
  for (kp = BufBndr[my_id][2]; kp <= BufBndr[my_id][5]; kp++) {     
    for (jp = BufBndr[my_id][1]; jp<=BufBndr[my_id][4]; jp++) {
      for (ip =BufBndr[my_id][0]; ip<=BufBndr[my_id][3]; ip++) {

/* #ifndef MPI_PARALLEL /\* if Not parallel *\/ */
/* 	ijkGlobToLoc(pG, ip, jp, kp, &iloc, &jloc, &kloc); */
/* 	pG->yglob[kp][jp][ip].ro = pG->U[kloc][jloc][iloc].d; */
/* #endif */

	//printf("NmaxArray= %d \n\n ", pG->GridOfRays[kp][jp][ip].len);

	//	printf("is, ie = %d %d id = %d \n", BufBndr[my_id][0], BufBndr[my_id][3], my_id);


       /* printf("ip = %d id = %d\n", ip, my_id); */
	       
	pG->yglob[kp][jp][ip].tau = 0.;
	tau=0;

	if (ip==is){
	  tau = pG->yglob[kp][jp][ip].tau;
	}
	
	for (ii = 0; ii<(pG->GridOfRays[kp][jp][ip]).len; ii++){

	  i=(pG->GridOfRays[kp][jp][ip]).Ray[ii].i;
	  j=(pG->GridOfRays[kp][jp][ip]).Ray[ii].j;
	  k=(pG->GridOfRays[kp][jp][ip]).Ray[ii].k;

	  // ijkGlobToLoc(pG, i, j, k, &iloc, &jloc, &kloc);	       			

	  //den = pG->U[kloc][jloc][iloc].d;
	  den= 	pG->yglob[k][j][i].ro;
	    
	  /* printf("%d %d %d %f \n", i,j,k, den); */

	  dl = (pG->GridOfRays[kp][jp][ip].Ray[ii]).dl;
	  //	printf("dl = %f \n\n", dl);
	  dtau = Rsc*Dsc*KPE* dl*den;

	  /* dtau = dl*den; */
	  tau+=dtau;
	  //	printf("dl = %f \n\n", dl);

	}
	pG->yglob[kp][jp][ip].tau = tau;
		
		/* printf("tau = %f %d %d %d \n",pG->yglob[kp][jp][ip].tau, ip,jp,kp ); */
      } //ip
    } //jp
  } //kp

}
void Constr_optDepthStackOnGlobGrid(MeshS *pM, GridS *pG){
  // it is assumed that the source is located on the axis of symmetry


   Real x1is, x2js, x3ks, den,  ri, tj, zk,x1,x2,x3,
   	   	   l,dl,tau=0,dtau=0 ;
   
   int i,j,k,is,ie,js,je,ks,ke,ip,jp,kp,iloc,jloc,kloc,
	knew,my_id=0,ii;

    Real abs_cart_norm, cart_norm[3], cyl_norm[3], xyz_src[3], xyz_pos[3], rtz_pos[3],rtz_in[3], xyz_p[3],
      res[1], olpos[3],xyz_cc[3], rtz_cc[3];
    int ijk_cur[3],iter,iter_max,lr,ir=0,i0,j0,k0,
    		NmaxArray=1;
    short nroot;
    Real xyz_in[3], radSrcCyl[3], dist, sint, cost, s2;
    
//    CellOnRayData arrayDataTmp; //indices, ijk and dl
//    int tmpIntArray1, *tmpIntArray2, *tmpIntArray3;

    CellOnRayData *tmpCellIndexAndDisArray;
    Real *tmpRealArray;

    short ncros;


 int  mpi1=1;

 is = 0;
 ie = pM->Nx[0]-1;     
 js = 0;
 je = pM->Nx[1]-1;       
 ks = 0;
 ke = pM->Nx[2]-1;

  
 iter_max= sqrt(pow(ie,2) + pow(je,2) +pow(ke,2));

//  allocating temporary array for 1D ray from point source
//  CellIndexAndCoords

  if ((tmpCellIndexAndDisArray=(CellOnRayData*)calloc(iter_max , sizeof(CellIndexAndCoords)))== NULL) {
      ath_error("[calloc_1d] failed to allocate memory CellIndexAndCoords\n");
    }

  cc_posGlobFromGlobIndx(pM, pG,is,js,ks, &x1is,&x2js,&x3ks);
  
  
  for (kp = ks; kp <= ke; kp++) { //z
    for (jp = js; jp <= je; jp++) { //t
      for (ip = is; ip <= ie; ip++) { //r

      #ifndef MPI_PARALLEL /* if Not parallel */
	ijkGlobToLoc(pG, ip, jp, kp, &iloc, &jloc, &kloc);
	pG->yglob[kp][jp][ip].ro = pG->U[kloc][jloc][iloc].d;
      #endif
	  
	  /* from ip, jp, kp get rtz position */
	cc_posGlobFromGlobIndx(pM,pG,ip,jp,kp,&rtz_pos[0],&rtz_pos[1],&rtz_pos[2]);
	   
	radSrcCyl[0] = 0.;
	radSrcCyl[1] = rtz_pos[1]; // corresponds to phi_jp
	radSrcCyl[2] =  0; // a ring at z=0

// 	   for parallel calls they are not the same with radSrcCyl
	rtz_in[0] = x1is;
	rtz_in[1] = rtz_pos[1]; // corresponds to phi_jp
	rtz_in[2] = rtz_pos[2] * rtz_in[0] / rtz_pos[0];

	/* Cart. position of the entry point */
	coordCylToCart(xyz_in, rtz_in, cos( rtz_in[1]), sin( rtz_in[1]));
	   	     
	/* Cart. position of the source point */
	coordCylToCart(xyz_src, radSrcCyl, cos(radSrcCyl[1]), sin(radSrcCyl[1]));
	    
	/* pG->disp[kp][jp][ip] =  sqrt(pow(rtz_in[0]-radSrcCyl[0],2)+ pow( rtz_in[2] 
	   - radSrcCyl[2],2)); */	    
	/* Cart. position of the starting point --> displacemnet*/

	/* indices of the starting point */
	lr=celli_Glob(pM,rtz_in[0], 1./pG->dx1, &i0, &ir, is);
	lr=cellj_Glob(pM,rtz_in[1], 1./pG->dx2, &j0, &ir, js);
	lr=cellk_Glob(pM,rtz_in[2], 1./pG->dx3, &k0, &ir, ks);

	ijk_cur[0] = i0;
	ijk_cur[1] = j0;
	ijk_cur[2] = k0;

// 	   printf("%f %d \n", x1is, i0); getchar();
//	    corrected location of the source on the grid
//		cc_pos(pG,i0,j0,k0,&rtz_pos[0],&rtz_pos[1],&rtz_pos[2]);

	sint = sin(rtz_pos[1]);
	cost = cos(rtz_pos[1]);
	/* from rtz position get get Cart position */
	coordCylToCart(xyz_p, rtz_pos, cost, sint);

	/* get normalized  vector from start to finish */
	for(i=0;i<=2;i++) cart_norm[i]= xyz_p[i]-xyz_src[i];

	abs_cart_norm = sqrt(pow(cart_norm[0], 2)+pow(cart_norm[1], 2)+pow(cart_norm[2], 2));

	dist = absValueVector(cart_norm);
	
	for(i=0;i<=2;i++) cart_norm[i] = cart_norm[i]/abs_cart_norm;
	cost = cos(radSrcCyl[1]);
	sint = sin(radSrcCyl[1]);
	cartVectorToCylVector(cyl_norm, cart_norm, cos(radSrcCyl[1]), sin(radSrcCyl[1]));
	/* tau = pG->tau_e[ kp ][ jp ][ 1 ]; */
	/* pG->tau_e[kp][jp] [ is ] = tau; */
	
	/* starting point for ray-tracing */
	rtz_pos[0]=rtz_in[0];
	rtz_pos[1]=rtz_in[1];
	rtz_pos[2]=rtz_in[2];
	sint = sin(rtz_pos[1]);
	cost = cos(rtz_pos[1]);
  
	for(i=0;i<=2;i++){
	  xyz_pos[i]=xyz_in[i];
	  xyz_cc[i]=xyz_in[i];
	}
		
	l = 0;
	dl =0;
	tau = 0;
	dtau=0;

	for (iter=0; iter<=iter_max; iter++){

	  s2 = pow(xyz_p[0]-xyz_pos[0],2)+pow(xyz_p[1]-xyz_pos[1],2)+pow(xyz_p[2]-xyz_pos[2],2);
	  olpos[0]=xyz_cc[0];
	  olpos[1]=xyz_cc[1];
	  olpos[2]=xyz_cc[2];
		  
	  traceGridCellOnGlobGrid(pM, pG, res, ijk_cur, xyz_pos, rtz_pos,
				  cart_norm, cyl_norm, &ncros);
	  /* find new cyl coordinates: */
	  cc_posGlobFromGlobIndx(pM,pG,ijk_cur[0],ijk_cur[1],ijk_cur[2],
				 &rtz_pos[0],&rtz_pos[1],&rtz_pos[2]);

	  ri = sqrt(pow(xyz_pos[0],2)+pow(xyz_pos[1],2));
	  tj = atan2(xyz_pos[1], xyz_pos[0]);
	  zk =xyz_pos[2];

	  /* we need only sign(s) of cyl_norm */
	  cyl_norm[0] = cart_norm[0]*cos(tj) + cart_norm[1]*sin(tj);
	  cyl_norm[1] = -cart_norm[0]*sin(tj) + cart_norm[1]*cos(tj);

	  dl = res[0];

	  rtz_cc[0]=rtz_pos[0];
	  rtz_cc[1]=rtz_pos[1];
	  rtz_cc[2]=rtz_pos[2];
		  
	  coordCylToCart(xyz_cc, rtz_cc,  cos(rtz_cc[1]),  sin(rtz_cc[1]) );
		  
	  dl = sqrt(pow(xyz_cc[0]-olpos[0],2)+pow(xyz_cc[1]-olpos[1],2)+
		    pow(xyz_cc[2]-olpos[2],2));

	  /* dl = sqrt( pow(pG->dx1,2)  +  pow(pG->dx3,2) ); */

	  (tmpCellIndexAndDisArray[iter]).dl = dl;
	  (tmpCellIndexAndDisArray[iter]).i = ijk_cur[0];
	  (tmpCellIndexAndDisArray[iter]).j = ijk_cur[1];
	  (tmpCellIndexAndDisArray[iter]).k = ijk_cur[2];

	  l += dl;
	  
/* #ifdef MPI_PARALLEL */
/* 	  den = (pG->yglob[ijk_cur[2]] [ijk_cur[1]] [ijk_cur[0]].ro < 1.05* rho0) ? */
/* 	    tiny : pG->yglob[ijk_cur[2]][ijk_cur[1]][ijk_cur[0]].ro; */
/* #else */
/* 	  den = (pG->U[ijk_cur[2]][ijk_cur[1]][ijk_cur[0]].d < 1.05* rho0) ? */
/* 	    tiny : pG->U[ijk_cur[2]][ijk_cur[1]][ijk_cur[0]].d;		  */
		    
/* #endif */
		  
/* 	  dtau = dl * den; */

/* 	  tau  += dtau; */
		  
	  NmaxArray = iter+1; //To use in allocation

	  //		  tau = l;
	  if (pow(xyz_p[0]-xyz_pos[0],2)+pow(xyz_p[1]-xyz_pos[1],2)+
	      pow(xyz_p[2]-xyz_pos[2],2)>s2){
	    //			tau= 0;
	    break;
	  }
		 

	  /* test if reached to the ip,jp,kp */
	  if( (ijk_cur[0]==ip &&  ijk_cur[1]==jp &&  ijk_cur[2]==kp)
	      /* || (ijk_cur[]==ip &&  ijk_cur[1]==jp &&  ijk_cur[2]==kp) */
	      ){

	    // cc_pos(pG, ijk_cur[0], ijk_cur[1], ijk_cur[2], &x1, &x2, &x3);

	    break;
	  }		  
	} //iter

	//pG->GridOfRays should be already allocated at init_grid.c
	(pG->GridOfRays[kp][jp][ip]).Ray =
	  (CellOnRayData*)calloc_1d_array(NmaxArray,sizeof(CellOnRayData));

	(pG->GridOfRays[kp][jp][ip]).len = NmaxArray;

	pG->tau_e[kp][jp][ip] = 0;


	for (ii = 0; ii<NmaxArray; ii++){

	  (pG->GridOfRays[kp][jp][ip]).Ray[ii].dl= (tmpCellIndexAndDisArray[ii]).dl;
	  (pG->GridOfRays[kp][jp][ip]).Ray[ii].i=(tmpCellIndexAndDisArray[ii]).i;
	  (pG->GridOfRays[kp][jp][ip]).Ray[ii].j=(tmpCellIndexAndDisArray[ii]).j;
	  (pG->GridOfRays[kp][jp][ip]).Ray[ii].k=(tmpCellIndexAndDisArray[ii]).k;

	  //		   printf("NmaxArray= %d dl \n\n ", NmaxArray);
	
	  /* pG->tau_e[kp][jp][ip] += dl; */
	}
	
	
	/* for (ii = 0; ii<(pG->GridOfRays[kp][jp][ip]).len; ii++){ */
	/* 	i=(pG->GridOfRays[kp][jp][ip]).Ray[ii].i; */
	/* 	j=(pG->GridOfRays[kp][jp][ip]).Ray[ii].j; */
	/* 	k=(pG->GridOfRays[kp][jp][ip]).Ray[ii].k; */
	/* 	dl = (pG->GridOfRays[kp][jp][ip].Ray[ii]).dl; */
			
	/* 	dtau = dl; */
	/* 	tau+=dtau; */
	/* } */

	

		      
		  
	//		pG->tau_e[kp][jp][ip] *= Rsc*Dsc*KPE;
	//		for (ii = 0; ii<(pG->GridOfRays[kp][jp][ip]).len; ii++){
	//			pG->tau_e[kp][jp][ip] += (pG->GridOfRays[kp][jp][ip].Ray[ii]).dl;
	//		}

	//pG->yglob[kp][jp][ip].tau  =  tau; //pG->tau_e[kp][jp][ip];

	tau=0;


      } //ip

      
    } //jp
  } //kp

  /* for (j = js; j <=je; j++) printf("%f \n",  pG->yglob[ks][j][is].tau  ); */


free(tmpCellIndexAndDisArray);

//  printf(" optDepthStack \n");
//  getchar();
}

#ifdef XRAYS
void Constr_optDepthStack(MeshS *pM, GridS *pG){
  // it is assumed that the source is located on the axis of symmetry

//   Real r, t, z,
//    tau_ghost,xi=0,;

   Real x1is, x2js, x3ks, den,  ri, tj, zk,x1,x2,x3,
   	   	   l,dl,tau=0,dtau=0 ;
   
    int i,j,k,is,ie,js,je,ks,ke, il, iu, jl,ju,kl,ku,ip,jp,kp,
	knew,my_id=0,ii;

    Real abs_cart_norm, cart_norm[3], cyl_norm[3], xyz_src[3], xyz_pos[3], rtz_pos[3],rtz_in[3], xyz_p[3],
      res[1], olpos[3],xyz_cc[3], rtz_cc[3];
    int ijk_cur[3],iter,iter_max,lr,ir=0,i0,j0,k0,
    		NmaxArray=1;
    short nroot;

    Real xyz_in[3], radSrcCyl[3], dist, sint, cost, s2;
    
//    CellOnRayData arrayDataTmp; //indices, ijk and dl
//    int tmpIntArray1, *tmpIntArray2, *tmpIntArray3;

    CellOnRayData *tmpCellIndexAndDisArray;
	Real *tmpRealArray;

    short ncros;


//infinite loop for parallel debugging

int  mpi1=1;
// while( mpi1==1 );

#endif

  is = pG->is;
  ie = pG->ie;
  js = pG->js;
  je = pG->je;
  ks = pG->ks;
  ke = pG->ke;
  il = is - nghost*(ie > is);
  jl = js - nghost*(je > js);
  kl = ks - nghost*(ke > ks);
  iu = ie + nghost*(ie > is);
  ju = je + nghost*(je > js);
  ku = ke + nghost*(ke > ks);

  iter_max= sqrt(pow(ie,2) + pow(je,2) +pow(ke,2));

//  allocating temporary array for 1D ray from point source
//  CellIndexAndCoords

  if ((tmpCellIndexAndDisArray=(CellOnRayData*)calloc(iter_max , sizeof(CellIndexAndCoords)))== NULL) {
      ath_error("[calloc_1d] failed to allocate memory CellIndexAndCoords\n");
    }

  cc_pos(pG,is,js,ks,&x1is,&x2js,&x3ks);
  
  
  for (kp = ks; kp<=ke; kp++) { //z
    for (jp = js; jp<=je; jp++) { //t
      for (ip = is; ip<=ie; ip++) { //r

	#ifdef DBG_MPI_OPT_STACK
	  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
	  /* int  mpi1=1; */
	  /* if (my_id == 1) while( mpi1==1 ); */
	  
	#endif
	  
	   /* from ip, jp, kp get rtz position */
	   cc_pos(pG,ip,jp,kp,&rtz_pos[0],&rtz_pos[1],&rtz_pos[2]);

	   /* cc_posGlob(pM, pG, ip,jp,kp, &tmp[0],&tmp[1],&tmp[2]); */
	   /* printf("%f %f %f %f %f %f \n", rtz_pos[0], rtz_pos[1],rtz_pos[2], tmp[0],tmp[1],tmp[2]); *\/ */
	   /* getchar(); */
	   
           radSrcCyl[0] = 0.;
 	   radSrcCyl[1] = rtz_pos[1]; // corresponds to phi_jp
 	   radSrcCyl[2] =  0; // a ring at z=0

// 	   for parallel calls they are not the same with radSrcCyl
	   rtz_in[0] = x1is;
 	   rtz_in[1] = rtz_pos[1]; // corresponds to phi_jp

	   rtz_in[2] =  rtz_pos[2] * rtz_in[0] / rtz_pos[0];

	   /* Cart. position of the entry point */
 	    coordCylToCart(xyz_in, rtz_in, cos( rtz_in[1]), sin( rtz_in[1]));
	   	     
 	   /* Cart. position of the source point */
 	    coordCylToCart(xyz_src, radSrcCyl, cos(radSrcCyl[1]), sin(radSrcCyl[1]));
	    
	    /* pG->disp[kp][jp][ip] =  sqrt(pow(rtz_in[0]-radSrcCyl[0],2)+ pow( rtz_in[2] - radSrcCyl[2],2)); */	    
 	   /* Cart. position of the starting point --> displacemnet*/

 	   /* indices of the starting point */
 	   lr=celli(pG,rtz_in[0], 1./pG->dx1, &i0, &ir);
 	   lr=cellj(pG,rtz_in[1], 1./pG->dx2, &j0, &ir);
 	   lr=cellk(pG,rtz_in[2], 1./pG->dx3, &k0, &ir);
 	   ijk_cur[0] = i0;
 	   ijk_cur[1] = j0;
 	   ijk_cur[2] = k0;

// 	   printf("%f %d \n", x1is, i0); getchar();

//	    corrected location of the source on the grid
//		cc_pos(pG,i0,j0,k0,&rtz_pos[0],&rtz_pos[1],&rtz_pos[2]);

		sint = sin(rtz_pos[1]);
		cost = cos(rtz_pos[1]);
		/* from rtz position get get Cart position */
		coordCylToCart(xyz_p, rtz_pos, cost, sint);

		/* get normalized  vector from start to finish */
		for(i=0;i<=2;i++) cart_norm[i]= xyz_p[i]-xyz_src[i];

		abs_cart_norm = sqrt(pow(cart_norm[0], 2)+pow(cart_norm[1], 2)+pow(cart_norm[2], 2));

		dist = absValueVector(cart_norm);
	
		for(i=0;i<=2;i++) cart_norm[i] = cart_norm[i]/abs_cart_norm;
		cost = cos(radSrcCyl[1]);
		sint = sin(radSrcCyl[1]);
		cartVectorToCylVector(cyl_norm, cart_norm, cos(radSrcCyl[1]), sin(radSrcCyl[1]));
		/* tau = pG->tau_e[ kp ][ jp ][ 1 ]; */
		/* pG->tau_e[kp][jp] [ is ] = tau; */
	
		/* starting point for ray-tracing */
		rtz_pos[0]=rtz_in[0];
		rtz_pos[1]=rtz_in[1];
		rtz_pos[2]=rtz_in[2];
		sint = sin(rtz_pos[1]);
		cost = cos(rtz_pos[1]);
  
		for(i=0;i<=2;i++) xyz_pos[i]=xyz_in[i];
		l = 0;
		dl =0;
		tau = 0;
		dtau=0;

		for (iter=0; iter<=iter_max; iter++){

		  s2 = pow(xyz_p[0]-xyz_pos[0],2)+pow(xyz_p[1]-xyz_pos[1],2)+pow(xyz_p[2]-xyz_pos[2],2);
		  olpos[0]=xyz_cc[0];
		  olpos[1]=xyz_cc[1];
		  olpos[2]=xyz_cc[2];
		  
		  traceGridCell(pG, res, ijk_cur, xyz_pos, rtz_pos,
				cart_norm, cyl_norm, &ncros);
		  /* find new cyl coordinates: */
		  cc_pos(pG,ijk_cur[0],ijk_cur[1],ijk_cur[2],&rtz_pos[0],&rtz_pos[1],&rtz_pos[2]);

		  ri = sqrt(pow(xyz_pos[0],2)+pow(xyz_pos[1],2));
		  tj = atan2(xyz_pos[1], xyz_pos[0]);
		  zk =xyz_pos[2];

		  /* we need only sign(s) of cyl_norm */
		  cyl_norm[0] = cart_norm[0]*cos(tj) + cart_norm[1]*sin(tj);
		  cyl_norm[1] = -cart_norm[0]*sin(tj) + cart_norm[1]*cos(tj);

		  dl = res[0];

		  rtz_cc[0]=rtz_pos[0];
		  rtz_cc[1]=rtz_pos[1];
		  rtz_cc[2]=rtz_pos[2];
		  
		  coordCylToCart(xyz_cc, rtz_cc,  cos(rtz_cc[1]),  sin(rtz_cc[1]) );
		  
		  dl = sqrt(pow(xyz_cc[0]-olpos[0],2)+pow(xyz_cc[1]-olpos[1],2)+
		     pow(xyz_cc[2]-olpos[2],2));

		  /* dl = sqrt( pow(pG->dx1,2)  +  pow(pG->dx3,2) ); */

		  (tmpCellIndexAndDisArray[iter]).dl = dl;
		  (tmpCellIndexAndDisArray[iter]).i = ijk_cur[0];
		  (tmpCellIndexAndDisArray[iter]).j = ijk_cur[1];
		  (tmpCellIndexAndDisArray[iter]).k = ijk_cur[2];

		  l += dl;

		  den = (pG->U[ijk_cur[2]][ijk_cur[1]][ijk_cur[0]].d < 1.05* rho0) ?
		    tiny : pG->U[ijk_cur[2]][ijk_cur[1]][ijk_cur[0]].d;
		  dtau = dl * den;
/* dtau = dl; */
		  tau  += dtau;
		  NmaxArray = iter+1; //To use in allocation

//		  tau = l;
		  if (pow(xyz_p[0]-xyz_pos[0],2)+pow(xyz_p[1]-xyz_pos[1],2)+pow(xyz_p[2]-xyz_pos[2],2)>s2){
//			tau= 0;
			break;
		  }

		 
//		  printf("%f \n", sqrt(s2));
//		  printf("%f %f %d %d %d %d\n", tau, dl, iter, kp, jp, ip);

//		  if (tau > 12.) {
//					  printf("iter, ip, jp, kp, ijk: %d %d %d %d %d %d %d %f %f %f\n",
//							  iter,ip,jp,kp, ijk_cur[0], ijk_cur[1], ijk_cur[2], tau, dtau,res[0]);
//					  printf("xyz_dest= %f %f %f xyz_pos= %f %f %f \n", xyz_p[0],xyz_p[1],xyz_p[2],
//					  xyz_pos[0],xyz_pos[1],xyz_pos[2]);
//				      printf("--------\n");
//				      getchar();
//		  }

		  /* test if reached to the ip,jp,kp */
		  if( ijk_cur[0]==ip &&  ijk_cur[1]==jp &&  ijk_cur[2]==kp ){
			cc_pos(pG,ijk_cur[0], ijk_cur[1], ijk_cur[2], &x1,&x2,&x3);
			break;
		  }

		  
		} //iter

		//		pG->GridOfRays should be already allocated at init_grid.c
		(pG->GridOfRays[kp][jp][ip]).Ray = (CellOnRayData*)calloc_1d_array(NmaxArray, sizeof(CellOnRayData));
		(pG->GridOfRays[kp][jp][ip]).len = NmaxArray;

		pG->tau_e[kp][jp][ip] = 0;

		for (ii = 0; ii<NmaxArray; ii++){
			(pG->GridOfRays[kp][jp][ip]).Ray[ii].dl= (tmpCellIndexAndDisArray[ii]).dl;
			(pG->GridOfRays[kp][jp][ip]).Ray[ii].i=(tmpCellIndexAndDisArray[ii]).i;
			(pG->GridOfRays[kp][jp][ip]).Ray[ii].j=(tmpCellIndexAndDisArray[ii]).j;
			(pG->GridOfRays[kp][jp][ip]).Ray[ii].k=(tmpCellIndexAndDisArray[ii]).k;
			
		}


	
	
		      /* for (ii = 0; ii<(pG->GridOfRays[kp][jp][ip]).len; ii++){ */
		      /* 	i=(pG->GridOfRays[kp][jp][ip]).Ray[ii].i; */
		      /* 	j=(pG->GridOfRays[kp][jp][ip]).Ray[ii].j; */
		      /* 	k=(pG->GridOfRays[kp][jp][ip]).Ray[ii].k; */
		      /* 	dl = (pG->GridOfRays[kp][jp][ip].Ray[ii]).dl; */
			
 		      /* 	dtau = dl; */
		      /* 	tau+=dtau; */
		      /* } */

	

		      
		  
//		pG->tau_e[kp][jp][ip] *= Rsc*Dsc*KPE;
//		for (ii = 0; ii<(pG->GridOfRays[kp][jp][ip]).len; ii++){
//			pG->tau_e[kp][jp][ip] += (pG->GridOfRays[kp][jp][ip].Ray[ii]).dl;
//		}

		pG->tau_e[kp][jp][ip]  =  tau; //pG->tau_e[kp][jp][ip];

		tau=0;


      } //ip
    } //jp
  } //kp

//  for (kp = ks; kp <=ke; kp++) printf("%f \n", pG->tau_e[kp][js][is]);


free(tmpCellIndexAndDisArray);

//  printf(" optDepthStack \n");
//  getchar();
}


void ionizParam(const MeshS *pM, GridS *pG){
  int i,k,is,ie,js,je,ks,ke,ip,jp,kp,i_g, j_g, k_g;
  Real x1, x2, x3, xi;
  
  is = pG->is;
  ie = pG->ie;
  js = pG->js;
  je = pG->je;
  ks = pG->ks;
  ke = pG->ke;
 
  
   for (kp=ks; kp<=ke; kp++) {   // z
    for (jp=js; jp<=je; jp++) { // phi        
      for (ip = is; ip<=ie; ip++) { //R
	/* position on glob */
	ijkLocToGlob(pG, ip,jp,kp, &i_g, &j_g, &k_g);

	pG->tau_e[kp][jp][ip] = pG->yglob[k_g][j_g][i_g].tau;

	if ((pG->U[kp][jp][ip].d > tiny)){
	  cc_pos(pG,ip,jp,kp, &x1,&x2,&x3);
	  
	  xi = Lx / (Nsc* fmax(pG->U[ kp ][ jp ][ ip ].d , rho0))/
	    pow(Rsc,2)/fmax(pow(x1 - rRadSrc[0], 2) + pow(x3 - rRadSrc[1] ,2), tiny);
	  /* printf("%f\n", Nsc); */
	}
	
	else{
	  printf("quite possibly something is negative in opticalProps ");
	  /* apause(); */
	}
	
	pG->xi[kp][jp][ip]  = xi;

	/* printf("%f \n", pG->yglob[k_g][j_g][i_g].tau);  */
	
	pG->xi[kp][jp][ip]  *=  exp(-pG->yglob[k_g][j_g][i_g].tau);


      }
    }
   }


	
}
  


void optDepthFunctions(GridS *pG){
  // it is assumed that the source is located on the axis of symmetry

  //	GridS *pG=pM->Domain[0][0].Grid;

  Real r, t, z, dl, sToX, sToZ,
    tau=0,dtau=0,tau_ghost,xi=0,x1,x2,x3, ro,rad,
    nnorm, norm[2], sfnorm[2],
    rCur[2],rNextX[2],rNextZ[2],colDens,
    den, xBndMax,zBndMin,zBndMax,x_is,z_is,res, sToX1, sToZ1, rsph;

  int i,k,is,ie,js,je,ks,ke, il, iu, jl,ju,kl,ku,ip,jp,kp,knew,
    my_id=0;
  
  float sgnf=1.;
  
  is = pG->is;
  ie = pG->ie;
  js = pG->js;
  je = pG->je;
  ks = pG->ks;
  ke = pG->ke;
  
  il = is - nghost*(ie > is);
  jl = js - nghost*(je > js);
  kl = ks - nghost*(ke > ks);
  
  iu = ie + nghost*(ie > is);
  ju = je + nghost*(je > js);
  ku = ke + nghost*(ke > ks);

#ifdef MPI_PARALLEL
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

  /* printf("optdepth,  my_id = %f \n",(float)my_id); */

  //infinite loop for parallel debugging
  int  mpi1=1;
  // while( mpi1==1 );
#endif
  
    
  for (kp=ks; kp<=ke; kp++) {   // z
    for (jp=js; jp<=je; jp++) { // phi

      tau = pG->tau_e[ kp ][ jp ][ 1 ];
     
      
      pG->tau_e[kp][jp] [ is ] = tau;
      
      // tau = 0;
      for (ip = is; ip<=ie; ip++) { //R

	//	pG->tau_e[kp][jp][ip] =  1.; //pG->tau_e[kp][jp][1];
	
	// Printf(" %d %d %d %d %d %d \n",il, jl, kl, iu,  ju,  ku );
	// getchar();
	//	tau=0.;
	//pG->tau_e[kp][jp][ip]=0.;
       
	cc_pos(pG,ip,jp,kp,&x1,&x2,&x3);

	nnorm = sqrt(pow(x1-rRadSrc[0], 2)+pow( x3-rRadSrc[1], 2));
	norm[0]=(x1-rRadSrc[0])/nnorm;
	norm[1]=(x3-rRadSrc[1])/nnorm;
	
	colDens =0.;
	x_is = pG->MinX[0] + 0.5*pG->dx1;
	
	sfnorm[0] =  copysign(1., norm[0] )*fmax(fabs(norm[0]),tiny);
	z_is = norm[1]/sfnorm[0]*(x_is - rRadSrc[0]) + rRadSrc[1];
      
	/* z_is = norm[1]/ fmax(norm[0],tiny)*(x_is - rRadSrc[0]) + rRadSrc[1]; */

	
	rCur[0]=x_is;
	rCur[1]=z_is;
	xBndMax = pG->MaxX[0];
      	zBndMin = pG->MinX[2];
      	zBndMax = pG->MaxX[2];
	
	rsph =  sqrt(pow(rCur[0] - rRadSrc[0], 2) + pow(rCur[1] - rRadSrc[1] ,2));

	//	tau = 0.;

	k = (int) ((z_is - pG->MinX[2])/pG->dx3 + pG->ks);

	i = is;
	while (i < ie)	{
	  sfnorm[0] =  copysign(1., norm[0] )*fmax(fabs(norm[0]),tiny);
	  sfnorm[1] =  copysign(1., norm[1] )*fmax(fabs(norm[1]),tiny);
	    
	  // crossing face, xf,i+1
	  rNextX[0] = pG->MinX[0] + (i+1 - pG->is)*pG->dx1;

	  rNextX[1]= norm[1]/sfnorm[0]*(rNextX[0]-rRadSrc[0])+rRadSrc[1];

	  if( rNextX[0]>pG->MaxX[0] ||  rNextX[1]>pG->MaxX[2] ||
	      rNextX[1]<pG->MinX[2] ){
	    break;
	  }

	  sToX = sqrt(pow(rNextX[0] - rCur[0], 2) + pow(rNextX[1] - rCur[1] ,2));
		       
	  sToX1 = sqrt(pow(rNextX[0] - rRadSrc[0], 2) + pow(rNextX[1] - rRadSrc[1] ,2));

	  knew = k + ( copysign(1., norm[1]) );
			
	  rNextZ[1] = pG->MinX[2] + (  (Real) (knew - pG->ks)  )*pG->dx3;

	  if (norm[1] != 0){
	    rNextZ[0] =(norm[0]/sfnorm[1]) * ( rNextZ[1]-rRadSrc[1] )+rRadSrc[0];
	  } else{
	    rNextZ[0] =norm[0]/sfnorm[1]*(rNextZ[1]-rRadSrc[1])+rRadSrc[0];
	  }

	  sToZ = sqrt(pow(rNextZ[0] - rCur[0], 2) + pow(rNextZ[1] - rCur[1] ,2));
	  sToZ1 = sqrt(pow(rNextZ[0] - rRadSrc[0], 2) + pow(rNextZ[1] - rRadSrc[1] ,2));

	  dl=0.;
	  if (sToZ>sToX){
	    // Lz>Lx, k is the same
	    dl = sToX;
	    i+=1;
	    rCur[0]=rNextX[0];
	    rCur[1]=rNextX[1];
	  } else{
	    //Lx>=Lz
	    dl = sToZ; //up or down
	    k=knew;
	    rCur[0]=rNextZ[0];
	    rCur[1]=rNextZ[1];
	  }
	  den = (pG->U[k][jp][i].d < 1.05* rho0) ? tiny : pG->U[k][jp][i].d;

	  den=1.;

	  
	  dtau = dl * den;

	  tau  += dtau;
	  
	  // printf("%f %d, %d , %d , %d  \n", tau, i ,ip,jp,kp );
	  
	  if ( k >= ke || k<ks || i==ip ){
	    //	    printf("%f \n", pG->tau_e[kp][jp][ip] );
	    break;
	  }



	  
	} /* i-loop:  -- end of ray tracing -- */
	
	

	//pG->tau_e[kp][jp][ip] =pG->tau_e[kp][jp][0]*(float)my_id;
	//printf("%f \n", pG->tau_e[kp][jp][ip] );
	//	Pg->tau_e[kp][jp][ip]=10;//(float)my_id;

	
	//test, using the spherical radius as a benchmark
	/*   	uncommnet below */
 	pG->tau_e[kp][jp][ip] = Rsc*KPE*Dsc*tau;
	rsph =  sqrt(pow(x1 - rRadSrc[0], 2) + pow(x3 - rRadSrc[1] ,2));

	pG->tau_e[kp][jp][ip] =  tau;

	pG->tau_e[kp][jp][ip] =  tau/rsph;//pG->tau_e[ kp ][ jp ][ 1 ];//1.;

	//	printf("%f %d %d %d \n", pG->tau_e[ kp ][ jp ][ 1 ], kp, jp, ip);
	
	
	if ((pG->U[k][jp][i].d > tiny) || (rsph > tiny)){

	  xi = Lx / (Nsc* fmax(pG->U[ kp ][ jp ][ ip ].d , tiny))/ pow( Rsc*rsph, 2 );

	}
	else{
	  printf("quite possibly something is negative in opticalProps ");
	  apause();
	}
	pG->xi[kp][jp][ip]  = xi;

	pG->xi[kp][jp][ip]  *=  exp(- (pG->tau_e[kp][jp][ip]) );

      } //ip = x

    }	// jp = phi
  }  //kp = z

}


Real updateEnergyFromXrayHeatCool(const Real E, const Real d,
				  const Real M1,const Real M2, const Real M3,
				  const Real B1c,const Real B2c,const Real B3c,
				  const Real xi, const Real dt,
				  int i, int j, int k)
{
  Real Pnew,dP, t_th,Hx, Cx,  dPmax,res,dHxdT, Tg, rts, xi_in, Press;
  int status;

  Real di = 1./d;

  Press = E -  0.5*(SQR(M1) + SQR(M2) + SQR(M3))*di;

  Press -=  0.5*(SQR(B1c) + SQR(B2c) + SQR(B3c));

  Press *=  Gamma_1;
  
  dPmax = 0.10;
  
  Real xi_min = 1;
  
  xi_in = fmax(xi, xi_min);

  if (d> 0.51){ 
    //printf("Dens, Press %e %e %d %d %d \n", d, Press, i, j, k);
  }
  
  if(isnan(Press)){
    printf("Press is nan %e %e \n", d, Press);
    return (0.);
  }
  
  xRayHeatCool(d, Press, xi_in, &Hx, &dHxdT, dt);
  
  t_th=Press/Gamma_1/fmax( fabs(Hx), tiny);
  
  Pnew = Press + dt* Hx/Gamma_1;


  dP = Pnew-Press;
  if( fabs(dP/Press ) > dPmax){
    dP =fabs(Press) * copysign(dPmax,  dP );
  }

  Hx = dP/(dt)/Gamma_1;

  /* if (Hx!=0) printf("%e %e %e %e %e\n", t_th, *dt, Hx, Press, xi_in); */

  if ( t_th < dt){

    Pnew = rtsafe_energy_eq(Press, d, xi_in, dt, &status);


    if (status==1) {
    
    /* printf("stop 2: %e %e res= %e t= %e H= %e stat= %d \n", Press, Pnew, res, t_th, Hx, status); */
    /* apause(); */

    }

    

  //		 if (dP >0) printf("%f \n", dP);

    Tg =  Pnew * Esc * M_MW / (d*Dsc*RGAS );

    if (Tg > Tx) Tg = Tx;

    /*   printf(" Tg = %e dens = %e \n", Tg, dens); */
    /* } */

    Pnew = Tg* (d*Dsc*RGAS)/(Esc * M_MW);
    dP = Pnew-Press;
    
    if( fabs(dP/Press ) > dPmax){
    

      dP =fabs(Press) * copysign(dPmax,  dP );

      //		  printf("%e %e %e %e %f \n",max_delta_e, eold,  dt*Hx, de/eold, Tx);

    }
  Hx = dP/(dt)/Gamma_1;

  }
  


  if(Pnew <= 0. ) Hx = 0.;

  //		 printf("negative pressure detected");


  //	 Hx = (abs(Hx)>0.0001 ) ?  copysign(0.0001, Hx) : Hx;
  //	 printf( " %e %e \n", Press, Pnew );
  //	 Tg = fabs( Press * Esc * M_MW) / (dens*Dsc*RGAS );

  /* if(xi_in>0.) printf("%e %e %e %e %e %e \n", Hx, U->d, Press,Pnew, Tg, xi_in); */

  	
  // Hx=0;
	 return(-Hx); /* returns cooling NOT  heating rate */

}

void xRayHeatCool(const Real dens, const Real Press, const Real xi_in,
		  Real *Hx, Real *dHxdT, const Real dt)
{
  Real max_delta_e = 0.05, Gx,Gc,Lbr,lbr1,lbr2,dGcdT,dGxdT,dLbrdT,dHxde,Tg,xi,dP,
    res=0.,
    epsP=0.5,
    delt=1.,

    c1 = 8.9e-36,
    c2 = 1.5e-21,
    al0 = 3.3e-27,
    al1 = 1.7e-18,
    al2 = 1.3e5,
    bta = -1.;
  Real const Tgmin = 10.;

  Tg = fabs( Press * Esc * M_MW) / (dens*Dsc*RGAS );

  /* if (Tg >10*Tx){ */
  /*   printf(" Tg = %e dens = %e \n", Tg, dens); */
  /* } */
	   
  //  return(Press/Gamma_1);
  //  egas = Evir0 * eij
  //  Tg = fabs( (gam -1.)*m_mw *egas/ (nd* nc0*m_u) /Rgas )

  xi = xi_in;

  //	xi = fmax(xi_in, 1.);

  //	 if(xi_in < 1.) {*Hx=0.; return;}

  Gc =  0.;
  Gx =  0.;
  Lbr = 0.;
  
  Gx= c2*pow(xi,0.25) /pow(Tg,0.5)*(1. - Tg/Tx);
  Gc = c1* xi*(Tx - 4.*Tg);

  lbr1= al0*sqrt(Tg);
  lbr2= ( al1*exp(-al2/Tg)*( pow(xi,bta) ) /sqrt(Tg)+1.e-24)*delt;

  /* if (Tg <= 1.e3 ){ */

  lbr2 =  ( al1*exp(-al2/Tgmin)*( pow(xi,bta) ) /sqrt(Tgmin)+1.e-24)*delt;

  /* } */

  
  Lbr= lbr1 + lbr2; //(erg cm^3/s)

  dP = dt* fabs(Lbr)/Gamma_1;
  if( fabs ( dP / Press ) > epsP) Lbr = fabs(Press) *epsP /dt/Gamma_1;

 //	 if(xi_in < 0.01) Lbr = 0.;

  *Hx = Gc + Gx - Lbr;

  
  
  dGcdT = - 4.*c1*xi;
  dGxdT = - 0.5*c2/Tx * pow(xi,0.25) *(Tx/Tg+1.)*pow (Tg, -0.5);
  
  dLbrdT = 0.5* al0 *pow(Tg,-0.5) + delt*al1*pow(xi,bta) *exp(-al2/Tg)*
    ( al2 -  0.5* Tg  )  * pow(Tg,-2.5);
  
  *dHxdT = dGcdT + dGxdT - dLbrdT;

  /* *Hx = Gc; */
  /* *dHxdT = dGcdT; */

  
  dHxde = *dHxdT*Gamma_1*M_MW*Evir0/Dsc/RGAS/Lamd0 * dens * pow( nc0, 2);

  *Hx = *Hx/Lamd0 * pow(dens*nc0, 2);

    
  //	 if(xi_in>100.)  printf("%e \n", *Hx );

  /* printf("Lamd0, pow(dens*nc0, 2),  %e %e \n\n", Lamd0, pow(dens*nc0, 2) ); */
  
  /* if(Tg > 1e4) { */
  /*   printf("Tg= %e Tx = %e \n", Tg, Tx); */
  /*   printf("*Hx, Lbr, Gc, Gx  %e %e %e %e xi= %e d= %e\n\n", *Hx, Lbr, Gc, Gx, xi, dens); //apause(); */
  /* } */
  /* printf("dens, Press, Tg, xi %e %e %e %e \n", dens, Press, Tg, xi); */

  if(isnan(*Hx)){
    
    printf("Hx or/and Cx is nan %e %e %e %e \n", dens, Press, Tg, xi);
    *Hx = 0.;
    
    return;
  }

  //	 *Hx = 0.;
  //	 printf("%e %e %e %e %e \n", Hx, Tg, xi, dens, Press);
}

#endif  /* XRAYS */


void plot(MeshS *pM, char name[16]){
  int i,j,k,is,ie,js,je,ks,ke,nx1,nx2,nx3,nr,nz, il, jl, kl, iu, ju, ku;
  char cwd[1024];
  getcwd(cwd, sizeof(cwd));
  
  GridS *pG=pM->Domain[0][0].Grid;
  
  printf("WORKING DIR: %s \n" ,  getcwd(cwd, sizeof(cwd) ) );

#ifdef MPI_PARALLEL
  FILE *f = fopen("./athena_plot.tmp", "w");
#else
  FILE *f = fopen("./athena_plot.tmp", "w");
#endif
  
  if (f == NULL){

    printf("Error opening file!\n");
    exit(1);
  }
  is = pG->is; ie = pG->ie;
  js = pG->js; je = pG->je;
  ks = pG->ks; ke = pG->ke;
  
  il = is - nghost*(ie > is);
  jl = js - nghost*(je > js);
  kl = ks - nghost*(ke > ks);
  
  iu = ie + nghost*(ie > is);
  ju = je + nghost*(je > js);
  ku = ke + nghost*(ke > ks);

  nr= (ie-is)+1;
  nz=(ke-ks)+1;
  nx1 = (ie-is)+1 + 2*nghost;
  nx2 = (je-js)+1 + 2*nghost;
  nx3 = (ke-ks)+1 + 2*nghost;

  //	printf("%d %d %d %d %d %d %d\n", nx1,nx2,nx3,nghost, (ie-is)+1 ,(je-js)+1,ie); apause();

  /* for(k=4;k<=10;k++) printf( "%f \n", pG->tau_e[k][10][20] ) ; */
  /* getchar(); */
 
  j=js;
  /* fprintf(f, "%d  %d\n", nr, nz ); */
  fprintf(f, "%d  %d\n", nx1, nx3 );
  for (k=kl; k<=ku; k++) {
    /* for (j=js; j<=je; j++){	  */
    for (i=il; i<=iu; i++){
      if (strcmp(name, "d") == 0){
	fprintf(f, "%f\n", pG->U[k][j][i].d );
      }
      if (strcmp(name, "E") == 0){
	fprintf(f, "%f\n", pG->U[k][j][i].E );
      }
#ifdef XRAYS
      if (strcmp(name, "tau") == 0){
	/* printf("here -------------------- %f, %d, %d, %d \n", */
	/*        pG->tau_e[k][j][i],  k,j,i); */
	
	fprintf(f, "%f\n", pG->tau_e[k][j][i] );
	
      }      
      if (strcmp(name, "xi") == 0){
	fprintf(f, "%f \n", pG->xi[k][j][i] );
      }
#endif
    }
    fprintf(f,"\n");
  }
  fclose(f);
  system("./plot_from_athena.py");
}

#ifdef XRAYS
void aplot(MeshS *pM, int is, int js, int ks, int ie, int je, int ke, char name[16]){
  
  int i,j,k,nx1,nx2,nx3,il, jl, kl, iu, ju, ku;

  GridS *pG=pM->Domain[0][0].Grid;
  char cwd[1024];
  getcwd(cwd, sizeof(cwd));
  
   
  printf("WORKING DIR: %s \n" ,  getcwd(cwd, sizeof(cwd) ) );

#ifdef MPI_PARALLEL
  FILE *f = fopen("./athena_plot.tmp", "w");
#else
  FILE *f = fopen("./athena_plot.tmp", "w");
#endif
  
  if (f == NULL){
    printf("Error opening file!\n");
    exit(1);
  }
   
  il = is;
  jl = js;
  kl = ks;
  
  iu = ie;
  ju = je;
  ku = ke;


  nx1 = (ie-is)+1;
  nx2 = (je-js)+1;
  nx3 = (ke-ks)+1;

  j=js;
  //printf(" %d %d \n", is, ie);
  
  fprintf(f, "%d  %d\n", nx1, nx3 );
  
  for (k=kl; k<=ku; k++) {
    for (i=il; i<=iu; i++){
      if (strcmp(name, "tau") == 0){
	fprintf(f, "%f\n",  pG->yglob[k][j][i].tau );
      }

      else if (strcmp(name, "ro") == 0){
	fprintf(f, "%f\n",  pG->yglob[k][j][i].ro );
      }
    }     
    fprintf(f,"\n");
  }
  
  fclose(f);
  system("./plot_from_athena.py");
}

#endif  /* XRAYS */


static Real grav_pot(const Real x1, const Real x2, const Real x3) {
  Real rad;


  rad = sqrt( SQR(x1) + SQR(x3) );

  return -1.0/(rad-rgToR0);

  // return 0.0;
}

/*! \fn static Real grav_acc(const Real x1, const Real x2, const Real x3)
 *  \brief Gravitational acceleration */

static Real grav_acc(const Real x1, const Real x2, const Real x3) {

  Real rad,res;

  rad = sqrt( SQR(x1) + SQR(x3) );

  res = (1.0/(SQR(rad-rgToR0)))*(x1/rad);

  return (res);

  //  return (1.0/(SQR(rad-2.0)))*(x1/rad); old

}

//Private functions (x1,x2,x3) = (R,p,z)

/*! \fn Real density(Real x1, Real x2, Real x3)
 *  \brief Calculates the density at x1, x2, x3*/

#define VP(R) pow(R/xm, 1.-q)

Real density(Real x1, Real x2, Real x3) {
  Real rad, tmp, d, res;
  rad = sqrt( SQR(x1) + SQR(x3));


#ifdef SOLOV
  tmp = (Ctor + xm/(rad-rgToR0)  - a1*pow(x1,-q1)) /(nAd+1)/Kbar;
  res= Br0_Sol + pow(x1,2)/2. + 1/sqrt(pow(x1,2) + pow(x3,2)) -
    (Br1_Sol + (w0_Sol*pow(x1,2))/2.)*(a2_Sol*pow(-1 + pow(x1,2),2) + (b1_Sol + a1_Sol*(-1 + pow(x1,2)))*pow(x3,2));
  res /= ((nAd+1)*Kbar);
  d = pow(res, nAd)*( res >=0.01 );
#endif


#ifdef HAWL
  tmp = (Ctor + (xm*1.0/( rad-rgToR0 )) - pow(xm, 2.0*q )*pow( x1, -q1 )/q1)/(nAd+1)/Kbar;
  d = pow(tmp, nAd)*( x1>=r_in );
#endif


  d = MAX(d, rho0);

  //  printf("in density()  %f %f %f %f %f %f\n", r_in, nAd, x1, rad, tmp, d); pause();
  //	  printf("%f %f %f %f %f %f\n", r_in, nAd, x1, rad, tmp, d); pause();
  //if (res>0.){
  //      printf("Kbar = %f %f\n", Kbar, Kbar_Sol);
  //  	  printf("%f %f %f %f\n", x1, tmp, res, d);
  // }
  //  d = pow(temp/Kbar,n)*(x1>=r_in);
  //  d = MAX(d,rho0);
  //  d = x1 < r_in ? rho0 : d;

  return d;

}

/*! \fn Real Volume(Grid *pG, int i, int j, int k)
 *  \brief Calculates the volume of cell (i,j,k) */
Real Volume(GridS *pG, int i, int j, int k) {
  Real x1,x2,x3;
  cc_pos(pG,i,j,k,&x1,&x2,&x3);
  // Volume_ijk = R_i * dR * dPhi * dZ
  return x1*pG->dx1*pG->dx2*pG->dx3;
}

VOutFun_t get_usr_out_fun(const char *name)
{
  return NULL;
}

/*----------------------------------------------------------------------------*/


/* problem:   */
//#define VP(R) (  (sqrt(r0)/(r0-2)) * pow(R/r0,-q+1)  )

//#define VP(R) pow(xm,q)*pow(R, 1.-q)


#define VP(R) pow(R/xm, 1.-q)


void problem(MeshS *pM, DomainS *pDomain)
{
  GridS *pG=(pDomain->Grid);

#ifdef XRAYS
  CoolingFunc = updateEnergyFromXrayHeatCool;
#endif


  int i,j,k;
  int is,ie,js,je,ks,ke,nx1,nx2,nx3;
  int il,iu,jl,ju,kl,ku;
  Real rad, IntE, KinE, MagE, rbnd, dz;
  Real x1,x2,x3, x1i, x2i, x3i,rhoa;
  Real Pgas, Pb, TotPgas, TotPb;
  Real scl, ***Ap, ***Bri, ***Bzi;
  Real divB=0.0, maxdivB=0.0;



  //  ------------------------------
 //  Soloviev solution

#ifdef SOLOV
  Real Psi_Sol;
#endif

  //  ------------------------------

  printf("problem \n");

  int my_id;
 
  
  is = pG->is; ie = pG->ie;
  js = pG->js; je = pG->je;
  ks = pG->ks; ke = pG->ke;
  nx1 = (ie-is)+1 + 2*nghost;
  nx2 = (je-js)+1 + 2*nghost;
  nx3 = (ke-ks)+1 + 2*nghost;
  il = is - nghost*(ie > is);
  jl = js - nghost*(je > js);
  kl = ks - nghost*(ke > ks);
  iu = ie + nghost*(ie > is);
  ju = je + nghost*(je > js);
  ku = ke + nghost*(ke > ks);


  if ((Ap = (Real***)calloc_3d_array(nx3+1, nx2+1, nx1+1, sizeof(Real))) == NULL) {
    ath_error("[HK-Disk]: Error allocating memory for vector pot\n");
  }

  if ((Bri = (Real***)calloc_3d_array(nx3+1, nx2+1, nx1+1, sizeof(Real))) == NULL) {
    ath_error("[HK-Disk]: Error allocating memory for Bri\n");
  }

  if ((Bzi = (Real***)calloc_3d_array(nx3+1, nx2+1, nx1+1, sizeof(Real))) == NULL) {
    ath_error("[HK-Disk]: Error allocating memory for Bzi\n");
  }

  /* Read initial conditions */
  F2Fedd = par_getd("problem","F2Fedd");
  fx = par_getd("problem","fx");
  fuv = par_getd("problem","fuv");
  nc0 = par_getd("problem","nc0");
  q      = par_getd("problem","q");
  r0     = par_getd("problem","r0");
  r_in   = par_getd("problem","r_in");
  rho0   = par_getd("problem","rho0");
  e0     = par_getd("problem","e0");
  seed   = par_getd("problem","seed");

#ifdef MHD
  dcut = par_getd("problem","dcut");
  beta = par_getd("problem","beta");
#endif
  
#ifdef RESISTIVITY

  eta_Ohm = 0.0;
  Q_AD    = 0.0;
  Q_Hall  = par_getd("problem","Q_H");
  //d_ind   = 1.0;


#endif


  //printf("%f \n", F2Fedd); getchar();


  Real Tg = fabs( Gamma_1*M_MW *e0/ (rho0*Dsc) /RGAS );

  //	printf(" %e \n", Tg); apause();

  calcProblemParameters();

  printProblemParameters();

  //    Ctor = pow(r_in,-q1)/q1 - 1.0/(r_in-rgToR0);
  //    Kbar = (Ctor + 1.0/(1.-rgToR0) - 1./q1) /(nAd+1);
  //    Ctor = pow(xm, 2.0*q) * pow(r_in,-q1)/q1 - xm*1.0/(r_in-rgToR0);
  //    Kbar = (Ctor + xm*1.0/(xm-rgToR0) - pow(xm, 2.0) /q1) /(nAd+1);


  //   	    cc_pos(pG, is+5,js,ks,&R_ib,&x2,&x3);
  //   	    cc_pos(pG, is,js,ks,&x1,&x2,&x3)	;
  //        printf(" %e  %e \n ", R_ib, x1); apause();
  //    	   Kbar_Sol = (C3_Sol + (6. + 3.*b1_Sol + a2_Sol*w0_Sol)/6.);
  //    	   Kbar_Sol /= (nAd+1);



  //    printf(" %e %e %e %e %e %e %e\n", q, Rsc, Dsc, Usc, Time0,Tvir0,Lamd0); apause();

  cc_pos(pG,is,js,ks,&x1,&x2,&x3);
  inRadOfMesh=x1;

  /* int  mpi1=1; */
  /* while( mpi1==1 ); */
 

  
  // Initialize data structures, set density, energy, velocity and Ap
  // Loop over all cells (including ghosts)
  for (k=kl; k<=ku; k++) {
    for (j=jl; j<=ju; j++) {
      for (i=il; i<=iu; i++) {

	// printf("%d\n", ju);
	// printf("%f \n", rho0); apause();

    	cc_pos(pG,i,j,k,&x1,&x2,&x3);
        rad = sqrt(SQR(x1) + SQR(x3));

        x1i = x1 - 0.5*pG->dx1;
        x2i = x2 - 0.5*pG->dx2;
        x3i = x3 - 0.5*pG->dx3;

        pG->U[k][j][i].d = rho0;
        pG->U[k][j][i].M1 = 0.0;


	pG->U[k][j][i].M2 = rho0 * sqrt(rad)/(rad-rgToR0);

	//        pG->U[k][j][i].M2 = VP(x1)*rho0;
	//        pG->U[k][j][i].M2 = 0.0;

        pG->U[k][j][i].M3 = 0.0;
        pG->U[k][j][i].E   = e0;

#ifdef MHD
        Ap[k][j][i] = 0.0;
        pG->B1i[k][j][i] = 0.0;
        pG->B2i[k][j][i] = 0.0;
        pG->B3i[k][j][i] = 0.0;
        pG->U[k][j][i].B1c = 0.0;
        pG->U[k][j][i].B2c = 0.0;
        pG->U[k][j][i].B3c = 0.0;
#endif

        // Set up torus

	//rbnd =1./xm/( pow(xm, 2.0*q) *pow(x1, -q1)/q1 - Ctor ) + rgToR0;

        rbnd =xm/( a1*pow(x1, -q1) - Ctor ) + rgToR0;

        if (rbnd<=0) { printf("rbnd<=0: %e \n", rbnd); apause();}

        r_in = 0.5;
	//        if ( (x1 >= r_in) && (rad <= rbnd) ) { //Checks to see if cell is in torus

	if (x1 >= r_in) { //Dummy

	  rhoa = density(x1, x2, x3);

	  if (rhoa>0.1){
	    //        		printf("%f \n", rhoa);
	  }

	  IntE = pow(rhoa, GAM53)*Kbar/Gamma_1;

	  // Add pressure fluctuations to seed instability
	  IntE = IntE*(1.0 - seed*sin(x2));

	  if ((IntE >= e0) && (rhoa >= rho0)) {
	    //If the values are above cutoff, set up the cell

            pG->U[k][j][i].d = rhoa;

	    /* pG->U[k][j][i].d = (Real)(my_id); */

	    /* if ( k > 20 ){ */

	    /*   pG->U[k][j][i].d =1.; */

	    /* } else{ */
       	
	    /* 	pG->U[k][j][i].d =0; */
	    /* } */


	    
            pG->U[k][j][i].M2 = VP(x1)*pG->U[k][j][i].d;

#ifdef SOLOV
	    Psi_Sol= a2_Sol*pow(-1 + pow(x1,2),2) + (b1_Sol + a1*(-1 + pow(x1,2)))*pow(x3,2);
	    pG->U[k][j][i].M2 = sqrt( ( 1.-w0_Sol* Psi_Sol  > 0 ) ) *pG->U[k][j][i].d;
#endif


            pG->U[k][j][i].E = IntE;

            //Note, at this point, E holds only the internal energy.  This must
            //be fixed later
          }

        }
      }
    }
  }

  // Calculate density and set up Ap if appropriate
  for (k=kl; k<=ku+1; k++) {
    for (j=jl; j<=ju+1; j++) {
      for (i=il; i<=iu+1; i++) {

	cc_pos(pG,i,j,k,&x1,&x2,&x3);

	rad = sqrt(SQR(x1) + SQR(x3));

	x1i = x1 - 0.5*pG->dx1;
        x2i = x2 - 0.5*pG->dx2;
        x3i = x3 - 0.5*pG->dx3;

        rhoa = density(x1i,x2i,x3i);

	//printf("%f \n", dcut); apause();

#ifdef MHD
        if (rhoa >= dcut) {
          // Ap = (density-cutoff)^2

          Ap[k][j][i] = SQR(rhoa-dcut);

        }
	//        else {
	//        	  Ap[k][j][i] = x1i;
	//        }

#endif //MHD

      }
    }
  }


#ifdef MHD
  // Set up interface magnetic fields by using Ap
  for (k=kl; k<=ku; k++) {
    for (j=jl; j<=ju; j++) {
      for (i=il; i<=iu; i++) {

        cc_pos(pG,i,j,k,&x1,&x2,&x3);
        x1i = x1 - 0.5*pG->dx1;
        x2i = x2 - 0.5*pG->dx2;
        x3i = x3 - 0.5*pG->dx3;

        // Br = -dAp/dz
        pG->B1i[k][j][i] = -(Ap[k+1][j][i]-Ap[k][j][i])/pG->dx3;

        // Bz = (1/R)*d(R*Ap)/dr
        pG->B3i[k][j][i] = (Ap[k][j][i+1]*(x1i+pG->dx1)-Ap[k][j][i]*x1i)/(x1*pG->dx1);

        //Bt = d(Ap)/dz - d(Ap)/dr
	//        pG->B1i[k][j][i] = 0.;
	//        pG->B3i[k][j][i] = 0.;

        dz =  copysign(pG->dx3, x3);

	//        pG->B2i[k][j][i] = fabs( ( Ap[k+1][j][i] - Ap[k][j][i])/dz  - (Ap[k][j][i+1] - Ap[k][j][i])/pG->dx1);

	//        if ( pG->B2i[k][j][i] != 0. ){
	//        		printf("%f %f\n", Ap[k+1][j][i],  Ap[k][j][i] ); // pG->B2i[k][j][i]);
	//        }


	//        // non-zero B in the empty space
	//	    cc_pos(pG,i,j,k,&x1,&x2,&x3);
	//	    rad = sqrt(SQR(x1) + SQR(x3));
	//	    x1i = x1 - 0.5*pG->dx1;
	//	    x2i = x2 - 0.5*pG->dx2;
	//    		x3i = x3 - 0.5*pG->dx3;
	//    		rhoa = density(x1i,x2i,x3i);
	//        if (rhoa <  dcut) {
	//        	pG->B3i[k][j][i] = (Ap[k][j][i+1]*(x1i+pG->dx1) - Ap[k][j][i]*x1i)/(x1*pG->dx1);
	//        pG->B3i[k][j][i] *= dcut;
	//        }

#ifdef SOLOV
	a1_Sol = 1.;
	a2_Sol = 1.;
	b1_Sol = 3.;
	pG->B1i[k][j][i] = (-2*(b1_Sol + a1_Sol*(-1 + pow(x1i,2)))*x3i)/x1i;   //Bx
	pG->B3i[k][j][i] =4*a2_Sol*(-1 + pow(x1i,2)) + 2*a1_Sol*pow(x3i,2); //Bz
#endif

      }
    }
  }
  //apause();
  // Calculate cell centered fields
  for (k=kl; k<=ku; k++) {
    for (j=jl; j<=ju; j++) {
      for (i=il; i<=iu; i++) {

        cc_pos(pG,i,j,k,&x1,&x2,&x3);

        if (i==iu)
          pG->U[k][j][i].B1c = pG->B1i[k][j][i];
        else

          pG->U[k][j][i].B1c = 0.5*((x1-0.5*pG->dx1)*pG->B1i[k][j][i]+(x1+0.5*pG->dx1)*pG->B1i[k][j][i+1])/x1;


        if (j==ju)
          pG->U[k][j][i].B2c = pG->B2i[k][j][i];
        else
          pG->U[k][j][i].B2c = 0.5*(pG->B2i[k][j+1][i] + pG->B2i[k][j][i]);

        if (k==ku)
          pG->U[k][j][i].B3c = pG->B3i[k][j][i];
        else
          pG->U[k][j][i].B3c = 0.5*(pG->B3i[k+1][j][i] + pG->B3i[k][j][i]);

	//        printf("B1c, B2c, B3c %E %E %E \n", pG->U[k][j][i].B1c, pG->U[k][j][i].B2c, pG->U[k][j][i].B3c);

      }
    }
  }
#endif //MHD


#ifdef MHD
  // Calculate scaling factor to satisfy beta, specifically Pgas and Pb per tile
  // Don't loop over ghosts
  Pgas = 0.0;
  Pb   = 0.0;

  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {

	Pgas += (Gamma-1)*pG->U[k][j][i].E*Volume(pG,i,j,k);

	Pb += 0.5*(SQR(pG->U[k][j][i].B1c) + SQR(pG->U[k][j][i].B2c)
		   + SQR(pG->U[k][j][i].B3c))*Volume(pG,i,j,k);
      }
    }
  }
#endif //MHD

#ifdef MPI_PARALLEL
  if(MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD, &my_id))
    ath_error("[main]: Error on calling MPI_Comm_rank in torus9.c\n");

  MPI_Reduce(&Pgas, &TotPgas, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&Pb, &TotPb, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
  if (my_id == 0) {
    printf("Total gas pressure = %f\n", TotPgas);
    printf("Total magnetic pressure = %f\n", TotPb);
  }
  
  MPI_Bcast(&TotPgas, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&TotPb, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

#else

  TotPgas = Pgas;
  TotPb = Pb;

#endif //PARALLEL


#ifdef MHD
  //calculate and use scaling factor so that the correct beta is ensured
  scl = sqrt(TotPgas/(TotPb*beta));

  printf("Using magnetic scaling factor %f\n", scl);
  for (k=kl; k<=ku; k++) {
    for (j=jl; j<=ju; j++) {
      for (i=il; i<=iu; i++) {
	
        pG->U[k][j][i].B1c *= scl;
        pG->U[k][j][i].B3c *= scl;
        pG->B1i[k][j][i]   *= scl;
        pG->B3i[k][j][i]   *= scl;

        cc_pos(pG,i,j,k,&x1,&x2,&x3);
	x1i = x1 - 0.5*pG->dx1;
	x2i = x2 - 0.5*pG->dx2;
	x3i = x3 - 0.5*pG->dx3;
	rhoa = density(x1i,x2i,x3i);

	//        if (rhoa <  dcut) {
	//        	printf("%f %f \n",
	//		pG->U[k][j][i].B1c,
	//		pG->U[k][j][i].B3c );
	//        }
      }
    }
  }
#endif


  // Fix E to hold the total energy (Kin + Int + Mag) (Loop over ghosts)
  for (k=kl; k<=ku; k++) {
    for (j=jl; j<=ju; j++) {
      for (i=il; i<=iu; i++) {

	KinE = 0.5*( SQR(pG->U[k][j][i].M1)+ SQR(pG->U[k][j][i].M2)
	      + SQR(pG->U[k][j][i].M3))/(pG->U[k][j][i].d);
        MagE = 0.0;

#ifdef MHD
        MagE = 0.5*(SQR(pG->U[k][j][i].B1c) + SQR(pG->U[k][j][i].B2c)
		    + SQR(pG->U[k][j][i].B3c));
#endif
        pG->U[k][j][i].E += KinE + MagE;
      }
    }
  }
  /* Enroll the gravitational function and radial BC */

  StaticGravPot = grav_pot;
  x1GravAcc = grav_acc;
  
  /* setting pointers to user BC */
  bvals_mhd_fun(pDomain, left_x1,  inX1BoundCond );
  bvals_mhd_fun(pDomain, left_x3,  diode_outflow_ix3 );
  bvals_mhd_fun(pDomain, right_x3,  diode_outflow_ox3);

  
#ifdef XRAYS
  /* optical depth is calculated for a given domain, boundary conditions are applied in */

  

#ifdef MPI_PARALLEL

  Constr_optDepthStackOnGlobGrid(pM, pG); //mainly calc. GridOfRays
  
  initGlobComBuffer(pM, pG);
  
  MPI_Barrier(MPI_COMM_WORLD);

  /* sync global patch */  
  
  SyncGridGlob(pM, &pDomain, pG, ID_DEN);
 
 /* aplot(pM, 0,0,0, pM->Nx[0]-1, pM->Nx[1]-1, pM->Nx[2]-1, "ro"); */

    /*  int  mpi1=1; */
    /* while(mpi1 == 1 ); */

  optDepthStackOnGlobGrid(pM, pG, my_id); //calc. tau 

  /* if (my_id ==3 ) { */
  /*   aplot(pM, 0,0,0, pM->Nx[0]-1, pM->Nx[1]-1, pM->Nx[2]-1, "tau"); */
  /* } */
  
  SyncGridGlob(pM, &pDomain, pG, ID_TAU);


  
     //  aplot(pM, 0,0,0, pM->Nx[0]-1, pM->Nx[1]-1, pM->Nx[2]-1, "tau");
 
    
  ionizParam(pM, pG);
  
#else  /* not parallel *cd/

    /* printf("  pM->RootMinX[0] = %f \n\n", pM->RootMinX[0] ); */


    Constr_optDepthStackOnGlobGrid(pM, pG); //mainly calc. GridOfRays

   
    optDepthStackOnGlobGrid(pM, pG); //calc. tau

    /* aplot(pM, 0,0,0, pM->Nx[0]-1, pM->Nx[1]-1, pM->Nx[2]-1, "tau"); */
 
    ionizParam(pM, pG);
        
    /* plot(pM, "xi"); */
  
    /* Constr_optDepthStack(pM, pG);     */
    /* optDepthStack(pM, pG); */
        
    /* plot(pM, "tau"); */
    /* plot(pM, "xi");  */

    
#endif /* MPI */
  
   
/* plot(pM, "xi"); */

//getchar();   
// 
   /* bvals_tau(pDomain); */
   /* getchar(); */

#endif





  //  for (k=kl; k<=ku; k++) {
  //    for (j=jl; j<=ju; j++) {
  //      for (i=il; i<=iu; i++) {
  //
  //		pG->U[k][j][i].d =0.1;
  //		pG->U[k][j][i].E  = e0;
  //
  //		pG->U[k][j][i].M1 = 0.;
  //
  //		cc_pos(pG,i,j,k,&x1,&x2,&x3);
  //
  //		pG->U[k][j][i].M2 = 1./x1;
  //
  ////				pG->U[k][j][i].M3 = 0.;
  //
  //      }
  //
  //    }
  //  }
  //  x1GravAcc = grav_acc;

  //  set_bvals_fun(left_x1,  disk_ir_bc);
  //  set_bvals_fun(right_x1,  disk_or_bc);
  //  set_bvals_fun(left_x3,  diode_outflow_ix3);
  //  set_bvals_fun(right_x3,  diode_outflow_ox3);


  //  plot(pG, "d");

  return;
}  //problem

/*==============================================================================
 * PROBLEM USER FUNCTIONS:
 * problem_write_restart() - writes problem-specific user data to restart files
 * problem_read_restart()  - reads problem-specific user data from restart files
 * get_usr_expr()          - sets pointer to expression for special output data
 * Userwork_in_loop        - problem specific work IN     main loop
 * Userwork_after_loop     - problem specific work AFTER  main loop
 * current() - computes x3-component of current
 * Bp2()     - computes magnetic pressure (Bx2 + By2)
 *----------------------------------------------------------------------------*/

//void problem_write_restart(GridS *pG, DomainS *pD, FILE *fp)

void problem_write_restart(MeshS *pM, FILE *fp)
{
  return;
}

void problem_read_restart(MeshS *pM, FILE *fp)
//void problem_read_restart(GridS *pG, DomainS *pD, FILE *fp)
{
  GridS *pG=pM->Domain[0][0].Grid;

  //  q = par_getd("problem","q");
  //  r0 = par_getd("problem","r0");
  //  r_in = par_getd("problem","r_in");
  //  rho0 = par_getd("problem","rho0");
  //  e0 = par_getd("problem","e0");
  //	seed = par_getd("problem","seed");

  //#ifdef MHD
  //  dcut = par_getd("problem","dcut");
  //  beta = par_getd("problem","beta");
  //#endif

  /* Enroll the gravitational function and radial BC */
  StaticGravPot = grav_pot;
  x1GravAcc = grav_acc;

  calcProblemParameters();
  printProblemParameters();

  bvals_mhd_fun(pG, left_x1,  inX1BoundCond );
  bvals_mhd_fun(pG, left_x3,  diode_outflow_ix3 );
  bvals_mhd_fun(pG, right_x3, diode_outflow_ox3);

  return;
}

ConsFun_t get_usr_expr(const char *expr)
{
  return NULL;
}

//Gasfun_t get_usr_expr(const char *expr)
//{
//  return NULL;
//}

//void Userwork_in_loop(Grid *pG, Domain *pDomain)  !A.D.

void Userwork_in_loop (MeshS *pM)
{

  int i,j,k,is,ie,js,je,ks,ke, nx1, nx2, nx3, il, iu, jl,ju,kl,ku;
  int prote, protd, my_id;
  int nl, nd;
  Real IntE, KinE, MagE=0.0, x1, x2, x3, DivB,Pgas,rad, dns,as_lim;

  Real di, v1, v2, v3, qsq,p, asq,b1,b2,b3, bsq,rho_max;

  static Real TotMass=0.0;

  //A.D.

  GridS *pG=pM->Domain[0][0].Grid;

  DomainS pD = pM->Domain[0][0];
	 
  is = pG->is; ie = pG->ie;
  js = pG->js; je = pG->je;
  ks = pG->ks; ke = pG->ke;
  nx1 = (ie-is)+1 + 2*nghost;
  nx2 = (je-js)+1 + 2*nghost;
  nx3 = (ke-ks)+1 + 2*nghost;
  il = is - nghost*(ie > is);
  jl = js - nghost*(je > js);
  kl = ks - nghost*(ke > ks);
  iu = ie + nghost*(ie > is);
  ju = je + nghost*(je > js);
  ku = ke + nghost*(ke > ks);

  // Verify divB
  protd = 0;
  prote = 0;

#ifdef MHD
  DivB = compute_div_b(pG);
#endif
  
  
#ifdef XRAYS

 
#ifdef MPI_PARALLEL

  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

  SyncGridGlob(pM, &pD, pG, ID_DEN);
 
  optDepthStackOnGlobGrid(pM, pG, my_id);
  
  SyncGridGlob(pM, &pD, pG, ID_TAU);  

  ionizParam(pM, pG);

  /* if (my_id == 1 ) { */
  /*   aplot(pM, 0,0,0, pM->Nx[0]-1, pM->Nx[1]-1, pM->Nx[2]-1, "tau"); */
  /*   aplot(pM, 0,0,0, pM->Nx[0]-1, pM->Nx[1]-1, pM->Nx[2]-1, "ro");     */
  /* } */
   
  
#else


   optDepthStackOnGlobGrid(pM, pG);

   ionizParam(pM, pG);
  
 /* aplot(pM, 0,0,0, pM->Nx[0]-1, pM->Nx[1]-1, pM->Nx[2]-1, "tau"); */
 /* plot(pM, "xi"); */

   /* optDepthStack(pM, pG); */
   /* plot(pM, "xi");    */
#endif /* parallel */

#endif
   
   

//plot(pM, "xi");

/* //for(k=4;k<=10;k++) printf("%f \n", pG->tau_e[k][10][10]) ; */
/* printf("optdepth after bvals,  my_id = %d \n",(int)my_id); */
/* if (my_id != 2) */
 /* change!  */
 /* bvals_tau(&pD); */
//plot(pM, "xi");
  //plot(pM, "E");


  for (k=kl; k<=ku; k++) {
    for (j=jl; j<=ju; j++) {
      for (i=il; i<=iu; i++) {

	cc_pos(pG,i,j,k,&x1,&x2,&x3);

	//        rad = sqrt(SQR(x1) + SQR(x3));
	//        if (rad< R_ib){
	//             pG->U[k][j][i].d = rho0;
	//             pG->U[k][j][i].E = e0;
	//             pG->U[k][j][i].M1= 0.;
	//             pG->U[k][j][i].M2= 0.;
	//             pG->U[k][j][i].M3= 0.;
	//        }


        KinE = 0.5*(SQR(pG->U[k][j][i].M1) + SQR(pG->U[k][j][i].M2)
		    + SQR(pG->U[k][j][i].M3))/(pG->U[k][j][i].d);
	
#ifdef MHD

        MagE = 0.5*(SQR(pG->U[k][j][i].B1c) + SQR(pG->U[k][j][i].B2c)
		    + SQR(pG->U[k][j][i].B3c));
#endif

        IntE = pG->U[k][j][i].E - KinE - MagE;
       
        if (isnan(pG->U[k][j][i].d) || isnan(pG->U[k][j][i].E)) {

	  printf("At pos isnan: (%d,%d,%d) (%f,%f,%f), Den1 = %f, E1 = %f\n", i, j, k, x1,x2,x3,
		 pG->U[k][j][i].d, pG->U[k][j][i].E);
	  printf("KinE1, MagE1, IntE (%f,%f,%f), \n", KinE,
		 MagE, IntE);
	  apause();

	  /* printf("%f %f %f \n", , );  */

          pG->U[k][j][i].d = rho0;
        } //end isnan check

	//        printf("%f %f\n", IntE, x1);
	//        printf("%f %f %f\n", pG->U[k][j][i].d, pG->U[k][j][i].M2, x1);

	//printf("%f %f \n", rho0, dcut); getchar();

        di = 1.0/(pG->U[k][j][i].d);
        v1 = pG->U[k][j][i].M1*di;
        v2 = pG->U[k][j][i].M2*di;
        v3 = pG->U[k][j][i].M3*di;
        qsq = v1*v1 + v2*v2 + v3*v3;
        b1 = pG->U[k][j][i].B1c
          + fabs((double)(pG->B1i[k][j][i] - pG->U[k][j][i].B1c));
        b2 = pG->U[k][j][i].B2c
          + fabs((double)(pG->B2i[k][j][i] - pG->U[k][j][i].B2c));
        b3 = pG->U[k][j][i].B3c
          + fabs((double)(pG->B3i[k][j][i] - pG->U[k][j][i].B3c));
        bsq = b1*b1 + b2*b2 + b3*b3;
        p = MAX(Gamma_1*(pG->U[k][j][i].E - 0.5*pG->U[k][j][i].d*qsq
			 - 0.5*bsq), TINY_NUMBER);
        asq = Gamma*p*di;

        rho_max = 100.;
        as_lim = 10.;

        //check for very low density
        pG->U[k][j][i].d = fmax(rho0,  pG->U[k][j][i].d );

        if( (sqrt(asq)>as_lim) || (pG->U[k][j][i].d > rho_max) ) {
	  dns = fmin(rho_max,  pG->U[k][j][i].d);

	  //check for very high density
	  pG->U[k][j][i].d = dns;

	  KinE = 0.5*(SQR(pG->U[k][j][i].M1) + SQR(pG->U[k][j][i].M2)
		      + SQR(pG->U[k][j][i].M3  ))/dns;
     
	  IntE = e0;  // Set this as well to keep c_s reasonable

	  pG->U[k][j][i].E = IntE + KinE + MagE;

	  protd++;
	  prote++;
        }
        else if (IntE < e0) {
          pG->U[k][j][i].E = e0 + KinE + MagE;
          prote++;
        }


        if (sqrt(asq)>as_lim){

	  printf("\n At pos (%d,%d,%d) (%f, %f, %f), Den1 = %e, E1 = %e\n", i, j, k, x1,x2,x3,
		 pG->U[k][j][i].d, pG->U[k][j][i].E);
	  printf("asq ,bsq, P, ro : %e %e %e  %e\n", asq, bsq, p, pG->U[k][j][i].d);
	  printf("%e %e", rho0, dcut);
        }



	//        else if (IntE > 1.) {
	//
	//          	pG->U[k][j][i].E = e0 + KinE + MagE;
	//
	//          	printf("torus9 d ,  IntE, KinE, MagE: %f %f %f %f\n", pG->U[k][j][i].d ,  IntE, KinE, MagE);
	//          	printf("At pos (%d,%d,%d) (%f,%f,%f), Den = %f, E = %f\n", i, j, k, x1,x2,x3,pG->U[k][j][i].d, pG->U[k][j][i].E);
	//          	prote++;
	//
	//        } //end  ro and E check


	//        IntE = pG->U[k][j][i].E - KinE - MagE;
	//        Pgas  = (Gamma-1)*IntE;
	//        xRayHeatCool(pG->U[k][j][i].d, Pgas, pG-> dt, pG->xi[k][j][i] );


	//      IntE = pG->U[k][j][i].E - KinE - MagE;
      }
    }
  }

#ifdef MPI_PARALLEL
 if (my_id == 0) {
    printf("\tDivergence @ Orbit %2.3f = %e\n",(pG->time)/61.6, DivB);
    if ((protd+prote) > 0) {
      printf("\tProtection enforced (D=%d,E=%d), Cumulative Mass = %2.5f\n", protd, prote,TotMass);
    }
  }
#else
 printf("\tDivergence @ Orbit %2.3f = %e\n",(pG->time)/61.6, DivB);
  if ((protd+prote) > 0) {
    printf("\tProtection enforced (D=%d,E=%d), Cumulative Mass = %2.5f\n", protd, prote,TotMass);

  }
#endif //PARALLEL

}  //end of Userwork_in_loop



void Userwork_after_loop(MeshS *pM)
{
	GridS *pG;
	DomainS *pD;
	int i,j,k,is,ie,js,je,ks,ke,kp,jp,ip;
	  is = pG->is;
	  ie = pG->ie;
	  js = pG->js;
	  je = pG->je;
	  ks = pG->ks;
	  ke = pG->ke;
#ifdef XRAYS
	  //	free some memory
	  for (kp = ks; kp<=ke; kp++) { //z
	    for (jp = js; jp<=je; jp++) { //t
	      for (ip = is; ip<=ie; ip++) { //r
	    	 free((pG->GridOfRays[kp][jp][ip]).Ray);
	      }
	    }
	  }
#endif
	  //	free_3d_array(pG->xi);
#ifdef MPI_PARALLEL
	  freeGlobArrays();
#endif	  
}

#define MAXIT 50
#define MAXIT_INTERV_EXPAND  5

#ifdef XRAYS
Real rtsafe_energy_eq(Real Press, Real dens, Real xi, Real dt, int* status)
{
  //returns new Pressure
  int j;
  Real df,dx,dxold,f,fh,fl,rat;
  Real temp,xh,xl,rts, x1, x2, x0,dHdT,Hx1,Hx2,
    xacc=1e-4;
  int i;

  x0 = Press;
  x1= Press;
  x2=x1;
  *status = 1;

  for (i =1; i<=MAXIT_INTERV_EXPAND; i++) {	/* loop 1 */
    rat  = 2.;
    x1 /= rat;
    x2 *= rat;
    
    xRayHeatCool(dens, x1, xi, &Hx1, &dHdT, dt);
    xRayHeatCool(dens, x2, xi, &Hx2, &dHdT, dt);

    fl = x1 - x0 -  dt* Hx1/Gamma_1;
    fh = x2 - x0 - dt* Hx2/Gamma_1;

    if (fl == 0.0) return x1;
    if (fh == 0.0) return x2;
    if (fl < 0.0) {
      xl=x1;
      xh=x2;
    } else {
      xh=x1;
      xl=x2;
    }

    if(fl*fh<0.) { /* in the interval  */
      //			printf("cond. met \n");
      *status=1;
      //			printf("%e %e \n", t_th, dt);
      //			apause();
      //			break;

      rts=0.5*(x1+x2);
      dxold=fabs(x2-x1);
      dx=dxold;

      xRayHeatCool(dens, rts, xi, &f, &df, dt);
      f =  rts - x0 -  dt* f/Gamma_1;
      df = 1. - df/Gamma_1;

      for (j=1;j<=MAXIT;j++) { /* loop 2 */

	//printf("iterations: %d \n", i);
	
	if ((((rts-xh)*df-f)*((rts-xl)*df-f) > 0.0)
	    || (fabs(2.0*f) > fabs(dxold*df))) {
	  dxold=dx;
	  dx=0.5*(xh-xl);
	  rts=xl+dx;
	  if (xl == rts) return rts;
	} else {
	  dxold=dx;
	  dx=f/df;
	  temp=rts;
	  rts -= dx;
	  if (temp == rts) return rts;
	}
	if (fabs(dx) < xacc) return rts;

	xRayHeatCool(dens, rts, xi, &f, &df, dt);
	f = rts - x0 - dt* f/Gamma_1;

	if (f < 0.0) {
	  xl=rts;
	} else {
	  xh=rts;
	}
      } /* loop 2 */

      /* Cap the number of iterations but don't fail */
      printf("iterations: %f \n", rts);
      /* Apause(); */
    }

    else{  /* not in the interval  */

      if ( i==MAXIT_INTERV_EXPAND ){
	status = 0;
	//				printf("cond. has not been met \n");
      }

    } /* interval check */

  } /* for loop 1 */

  return rts;

  //
  //	if ((fl > 0.0 && fh > 0.0) || (fl < 0.0 && fh < 0.0)) {
  //		ath_error("interval cannot be expanded further in torus9:rtsafe_energy_eq\n");

}
#endif


#undef MAXIT
#undef MAXIT_INTERV_EXPAND


static void inX1BoundCond(GridS *pGrid)
{
  int is = pGrid->is;
  int js = pGrid->js, je = pGrid->je;
  int ks = pGrid->ks, ke = pGrid->ke;
  int i,j,k;
  Real x1,x2,x3,rad,rsf,lsf;


#ifdef MHD
  int ju, ku; /* j-upper, k-upper */
#endif


  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=1; i<=nghost; i++) {


#ifdef show_debug_messages
	if( (pGrid->U[k][j][i].d < rho0) ||  (pGrid->U[k][j][i].d > 3.) ) {
	  printf("negative density at BC: %e \n", pGrid->U[k][j][i].d );
	}
#endif

	cc_pos(pGrid,is,j,k,&x1,&x2,&x3);

	//    	  printf("%f %f %f\n", pGrid->U[k][j][i].M1, pGrid->U[k][j][i].M2, pGrid->U[k][j][i].M3);
	//   	  printf("%d \n", nghost ); apause();

	rad = sqrt(SQR(x1) + SQR(x3));

	//       printf("%f %f\n", x1, x3);

	// printf("%d %d\n", nghost, is); getchar();

	pGrid->U[k][j][is-i] = pGrid->U[k][j][is];
	pGrid->U[k][j][is-i].M1 = MIN(pGrid->U[k][j][is-i].M1,0.0);

	//    	  	  if (pGrid->U[k][j][is].M1 > 0.) {
	//      	    		pGrid->U[k][j][is-i].M1 = 0.;
	//    	  	  }

	pGrid->U[k][j][is-i].M2 =  pGrid->U[k][j][is].d * sqrt(rad) / (rad-rgToR0);
	pGrid->U[k][j][is-i].M2 =  pGrid->U[k][j][is].d * sqrt(  1 / (rad-rgToR0 ) );




	//    	  pGrid->U[k][j][is-i].d =  fmax(rho0, pGrid->U[k][j][is].d) ;
	//    	      pGrid->U[k][j][is-i].d =  rho0;
	//    	      pGrid->U[k][j][is-i].E =  e0;

	//       printf("%f \n", pGrid->U[k][j][is-i].E);
      }
    }
  }

#ifdef MHD
  /* B1i is not set at i=is-nghost */
  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=1; i<=nghost-1; i++) {

	pGrid->B1i[k][j][is-i] = pGrid->B1i[k][j][is];

	rsf = (x1+0.5*pGrid->dx1);
	lsf = (x1-0.5*pGrid->dx1);

	pGrid->B1i[k][j][is-i] = pGrid->B1i[k][j][is]*rsf/lsf;


      }
    }
  }

  if (pGrid->Nx[1] > 1) ju=je+1; else ju=je;
  for (k=ks; k<=ke; k++) {
    for (j=js; j<=ju; j++) {
      for (i=1; i<=nghost; i++) {
        pGrid->B2i[k][j][is-i] = pGrid->B2i[k][j][is];

        pGrid->B2i[k][j][is-i] = 0;

      }
    }
  }

  if (pGrid->Nx[2] > 1) ku=ke+1; else ku=ke;
  for (k=ks; k<=ku; k++) {
    for (j=js; j<=je; j++) {
      for (i=1; i<=nghost; i++) {

        pGrid->B3i[k][j][is-i] = pGrid->B3i[k][j][is];

	//        pGrid->B3i[k][j][is-i] = 0;

      }
    }
  }
#endif /* MHD */

  return;
}


static void diode_outflow_ix3(GridS *pGrid)
{
  int is = pGrid->is, ie = pGrid->ie;
  int js = pGrid->js, je = pGrid->je;
  int ks = pGrid->ks;
  int i,j,k;

  for (k=1; k<=nghost; k++) {
    for (j=js-nghost; j<=je+nghost; j++) {
      for (i=is-nghost; i<=ie+nghost; i++) {
        pGrid->U[ks-k][j][i] = pGrid->U[ks][j][i];
        pGrid->U[ks-k][j][i].M3 = MIN(pGrid->U[ks-k][j][i].M3,0.0);
      }
    }
  }



#ifdef MHD
  /* B1i is not set at i=is-nghost */
  for (k=1; k<=nghost; k++) {
    for (j=js-nghost; j<=je+nghost; j++) {
      for (i=is-(nghost-1); i<=ie+nghost; i++) {
        pGrid->B1i[ks-k][j][i] = pGrid->B1i[ks][j][i];
      }
    }
  }

  /* B2i is not set at j=js-nghost */
  for (k=1; k<=nghost; k++) {
    for (j=js-(nghost-1); j<=je+nghost; j++) {
      for (i=is-nghost; i<=ie+nghost; i++) {
        pGrid->B2i[ks-k][j][i] = pGrid->B2i[ks][j][i];
      }
    }
  }

  /* B3i is not set at k=ks-nghost */
  for (k=1; k<=nghost-1; k++) {
    for (j=js-nghost; j<=je+nghost; j++) {
      for (i=is-nghost; i<=ie+nghost; i++) {
        pGrid->B3i[ks-k][j][i] = pGrid->B3i[ks][j][i];
      }
    }
  }
#endif /* MHD */

  return;
}

/*----------------------------------------------------------------------------*/
/*! \fn static void outflow_ox3(GridS *pGrid)
 *  \brief OUTFLOW boundary conditions, Outer x3 boundary (bc_ox3=2) */

static void diode_outflow_ox3(GridS *pGrid)
{
  int is = pGrid->is, ie = pGrid->ie;
  int js = pGrid->js, je = pGrid->je;
  int ke = pGrid->ke;
  int i,j,k;

  for (k=1; k<=nghost; k++) {
    for (j=js-nghost; j<=je+nghost; j++) {
      for (i=is-nghost; i<=ie+nghost; i++) {
        pGrid->U[ke+k][j][i] = pGrid->U[ke][j][i];

        pGrid->U[ke+k][j][i].M3 = MAX(pGrid->U[ke+k][j][i].M3,0.0);

      }
    }
  }

  //  pGrid->U[k][j][is-i].M1 = MIN(pGrid->U[k][j][is-i].M1,0.0);

#ifdef MHD
  /* B1i is not set at i=is-nghost */
  for (k=1; k<=nghost; k++) {
    for (j=js-nghost; j<=je+nghost; j++) {
      for (i=is-(nghost-1); i<=ie+nghost; i++) {
        pGrid->B1i[ke+k][j][i] = pGrid->B1i[ke][j][i];
      }
    }
  }

  /* B2i is not set at j=js-nghost */
  for (k=1; k<=nghost; k++) {
    for (j=js-(nghost-1); j<=je+nghost; j++) {
      for (i=is-nghost; i<=ie+nghost; i++) {
        pGrid->B2i[ke+k][j][i] = pGrid->B2i[ke][j][i];
      }
    }
  }

  /* k=ke+1 is not a boundary condition for the interface field B3i */
  for (k=2; k<=nghost; k++) {
    for (j=js-nghost; j<=je+nghost; j++) {
      for (i=is-nghost; i<=ie+nghost; i++) {
        pGrid->B3i[ke+k][j][i] = pGrid->B3i[ke][j][i];
      }
    }
  }
#endif /* MHD */

  return;
}




static void calcProblemParameters(){
  MSOLYR = 1.989e33/YR;
  SGMB = ARAD*CL/4;

  MBH_in_gram = M2Msun*MSUN;
  Rg = 2*GRV*MBH_in_gram/pow(CL,2);
  r0*=Rg;

  Nsc=nc0;
  Tx=1.0e8; //Compton temperature
  Tgmin = 2.3;

  Dsc = nc0*M_MW*	M_U;
  xm =1.;
  r_in *= xm;

  Rsc = xm* r0;
  rgToR0 = Rg/Rsc;
  //   	rgToR0 = 0.;

  //   	printf("Rsc = %e \n", Rsc); apause();

  Usc = sqrt(GRV*MBH_in_gram/Rsc);
  Time0=sqrt( pow(Rsc,3) /GRV/MBH_in_gram);
  Evir0=Dsc*GRV*MBH_in_gram/Rsc;  //(erg/cm^3)
  Psc = Evir0;
  Esc = Psc;

  Tvir0=GRV*MBH_in_gram/RGAS/Rsc*M_MW;
  Lamd0=Evir0/Time0;  // (erg/cm^3/s)
  Ledd_e = 4.*PI *CL*GRV*MBH_in_gram/KPE;
  Lx = fx * F2Fedd*Ledd_e;
  Luv= fuv *F2Fedd*Ledd_e;

  //  torus parameters

  nAd = 1.0/Gamma_1;
  q1 = 2.0*(q-1.0);

#ifdef HAWL

  a1 = pow(xm, q1) /q1;
  Ctor = a1* pow(r_in,-q1) - xm/(r_in-rgToR0);
  Kbar = (Ctor + xm/(xm-rgToR0) - 1./q1) /(nAd+1);


#endif

#ifdef SOLOV
  Kbar_Sol =(1.5 + Br0_Sol)/(1 + nAd);
  Kbar = Kbar_Sol;
#endif

  //    	printf( " %e %e %e %e\n", Dsc, nc0, M_MW, M_U);
  printf("Ctor, Kbar= %e %e \n", Ctor, Kbar );
}

static void printProblemParameters(){
  //	#ifdef SOLOV
  //	  printf("solov is not implemented"); getchar();
  //	#endif

  printf("parameters \n");

  printf("F2Fedd = %e \n", F2Fedd);
  printf("nc0 = %e \n", nc0);
  printf("R0 = %e \n", r0);
  printf("Ctor, Kbar= %e %e \n", Ctor, Kbar );

  //	getchar();
}


#ifdef testray1
void testRayTracings( MeshS *pM, GridS *pG){
    Real r, t, z, l,dl, ri, tj, zk,
    tau=0,dtau=0,tau_ghost,xi=0,x1,x2,x3;      
   
    int i,j,k,is,ie,js,je,ks,ke, il, iu, jl,ju,kl,ku,ip,jp,kp,knew,
    my_id=0;

    Real abs_cart_norm, cart_norm[3], cyl_norm[3], xyz_pos[3], rtz_pos[3], xyz_p[3],
      res[1];
    int ijk_cur[3],iter,iter_max, lr, ir=0, i0,j0,k0, i1,j1,k1;
    short nroot;

    Real xyz_in[3], radSrcCyl[3], tmp[3], dist, sint, cost;

       
  printf("hello from test ray-tracing \n");
  
  is = pG->is;
  ie = pG->ie;
  js = pG->js;
  je = pG->je;
  ks = pG->ks;
  ke = pG->ke;
  il = is - nghost*(ie > is);
  jl = js - nghost*(je > js);
  kl = ks - nghost*(ke > ks);
  iu = ie + nghost*(ie > is);
  ju = je + nghost*(je > js);
  ku = ke + nghost*(ke > ks);


  radSrcCyl[0] = rRadSrc[0];
  radSrcCyl[1] = 0.;
  radSrcCyl[2] = rRadSrc[1];

  /* start point */ 
  radSrcCyl[0]=0.01;
  radSrcCyl[1]=0.;
  radSrcCyl[2]=0.1;
  /* end point */
 
    
  x1 =  0.073083504695681381;
  x2 = -0.50423114872188946;
  x3 = -4.5;
  
  lr=celli(pG,radSrcCyl[0], 1./pG->dx1, &i0, &ir);
  lr=cellj(pG,radSrcCyl[1], 1./pG->dx2, &j0, &ir);
  lr=cellk(pG,radSrcCyl[2], 1./pG->dx3, &k0, &ir);

  ijk_cur[0] = i0;
  ijk_cur[1] = j0;
  ijk_cur[2] = k0;

  //  corrected location of the source on the grid
  cc_pos(pG,i0,j0,k0,&rtz_pos[0],&rtz_pos[1],&rtz_pos[2]);

  
  
  for(i=0;i<=2;i++) radSrcCyl[i]=rtz_pos[i];

  coordCylToCart(xyz_in, radSrcCyl, cos(radSrcCyl[1]), sin(radSrcCyl[1]) );

  //  corrected location of the end point on the grid
  lr=celli(pG, x1, 1./pG->dx1, &ip, &ir);
  lr=cellj(pG, x2, 1./pG->dx2, &jp, &ir);
  lr=cellk(pG, x3, 1./pG->dx3, &kp, &ir);    
  
  cc_pos(pG,ip,jp,kp,&rtz_pos[0],&rtz_pos[1],&rtz_pos[2]);
  sint = sin(rtz_pos[1]);
  cost = cos(rtz_pos[1]);
  coordCylToCart(xyz_p, rtz_pos, cost, sint);

  for(i=0;i<=2;i++) cart_norm[i]= xyz_p[i]-xyz_in[i];

  
  dist = absValueVector(cart_norm);
  abs_cart_norm = sqrt(pow(cart_norm[0], 2)+pow(cart_norm[1], 2)+pow(cart_norm[2], 2));
  for(i=0;i<=2;i++) cart_norm[i] = cart_norm[i]/ abs_cart_norm;  

  cartVectorToCylVector(cyl_norm, cart_norm, cos(radSrcCyl[1]), sin(radSrcCyl[1]));		    	     
 

  /* starting point for ray-tracing */
  rtz_pos[0]=radSrcCyl[0];
  rtz_pos[1]=radSrcCyl[1];
  rtz_pos[2]=radSrcCyl[2];
  sint = sin(rtz_pos[1]);
  cost = cos(rtz_pos[1]);
  
  for(i=0;i<=2;i++) xyz_pos[i]=xyz_in[i];

  iter_max = ke*je*ke;
  iter_max=50;
  
  for (iter=0; iter<=iter_max; iter++){
    
    traceGridCell(pG, res, ijk_cur, xyz_pos, rtz_pos,
		  cart_norm, cyl_norm, &nroot);

    /* /\* get a more accurate aim, re-target *\/ */
    /* normVectBToA(cart_norm, xyz_p, xyz_pos, &tmp); */
    
    /* find new cyl coordinates: */
    cc_pos(pG,ijk_cur[0],ijk_cur[1],ijk_cur[2],&rtz_pos[0],&rtz_pos[1],&rtz_pos[2]);

    ri = sqrt(pow(xyz_pos[0],2)+pow(xyz_pos[1],2));
    tj = atan2(xyz_pos[1], xyz_pos[0]);
    zk =xyz_pos[2];

    /* we need only sign(s) of cyl_norm */
    cyl_norm[0] = cart_norm[0]*cos(tj) + cart_norm[1]*sin(tj);
    cyl_norm[1] = -cart_norm[0]*sin(tj) + cart_norm[1]*cos(tj);

    /* get a non-normalized direction in cyl coordinates */
    //cartVectorToCylVector(cyl_norm, cart_norm, cos(rtz_pos[1]),sin(rtz_pos[1]));
    //printf("\n %f %f \n", cyl_norm[1], res[1]);
    
    // cartVectorToCylVector(cyl_norm, cart_norm, cos(rtz_pos[1]),sin(rtz_pos[1]));
    
    
    dl = res[0];
    l += dl;
    
//    printf("ip,jp,kp, ijk,l,dl: %d %d %d %d %d %d  %f %f \n",
//	   ip,jp,kp, ijk_cur[0], ijk_cur[1], ijk_cur[2],l, dl);
//
//    printf("xyz_dest= %f %f %f xyz_pos= %f %f %f \n", xyz_p[0],xyz_p[1],xyz_p[2],
//	   xyz_pos[0],xyz_pos[1],xyz_pos[2]);

 
    if( ijk_cur[0]==ip &&  ijk_cur[1]==jp &&  ijk_cur[2]==kp ){

      cc_pos(pG,ijk_cur[0], ijk_cur[1], ijk_cur[2], &x1,&x2,&x3);
      printf("compare: %f %f \n", l, dist);
      break;
    }
    /* s_prev = s; */    
    printf("iteration: %d \n", iter);       
   

  }
  
  printf(" test raytracing done ");
  getchar();
  return;
}

#endif  /* testray1 */

void traceGridCellOnGlobGrid(MeshS *pM,GridS *pG, Real *res, int *ijk_cur, Real *xyz_pos,
		   Real* rtz_pos, const Real *cart_norm, const Real *cyl_norm,			    
		   short* nroot) {
  /* ijk_cur is the 3d index of the current cell */
  /*  xpos is the (x,y,z) exact cart. position, should be located at one of the cell's boundaries; */
  /*  x1x2x3 is the 3d - r,t,z -center coordinates of the cell to which the 
      above boundaries belong */
  // traces a cingle cell; cnorm is in Cart. coordinates;

  int change_indx, icur, jcur, kcur, crosPhiConstPlanes=0;

  Real a,b,c, rfc, zfc, d,tfc,sint,cost,
    L=HUGE_NUMBER, L1=HUGE_NUMBER, L2=HUGE_NUMBER,
     xcur, ycur, zcur, one_over_a;
     int is,ie,js,je,ks,ke;
     *nroot=0;
     
     is = 0;
     ie = pM->Nx[0]-1;     
     js = 0;
     je = pM->Nx[1]-1;       
     ks = 0;
     ke = pM->Nx[2]-1;
  
     icur= ijk_cur[0]; /* --- icur etc. are on cyl. grid ----*/
     jcur= ijk_cur[1];
     kcur= ijk_cur[2];

     xcur = xyz_pos[0];   /*  are on Cart. grid */
     ycur=  xyz_pos[1];
     zcur=  xyz_pos[2];


      /*  1)  intersection with cylinders  */     
     
     if(cyl_norm[0]>0){       
       rfc = pM->RootMinX[0] + ((Real)(icur - is)+1.)*pG->dx1;
     }
     else if (cyl_norm[0]<0.){
       rfc = pM->RootMinX[0] + ((Real)(icur -  is))*pG->dx1;
     }
       else{ /* nr==0: almost never happens*/

	   goto z_planes;
     }
     change_indx = 0;
       
     a = pow(cart_norm[0],2) + pow(cart_norm[1],2);
     b = 2.*(xyz_pos[0]*cart_norm[0]+xyz_pos[1]*cart_norm[1]); 
     c = pow(xyz_pos[0],2)  + pow(xyz_pos[1],2) - pow(rfc,2);
     d = pow(b,2)-4.*a*c;

     if (d<0){ /* possibly at a grazing angle to a cylinder */

   #ifdef DEBUG_traceGridCell
    	 printf("Discr<0. in traceACell\n");
   #endif
       goto z_planes;
     }

      one_over_a =copysign(1./fmax(fabs(a), tiny), a);
      L1 = fabs(-b+sqrt(d))/2.*one_over_a;
      L2 = fabs(-b-sqrt(d))/2.*one_over_a;
      L = fmin(fabs(L1),fabs(L2));
      nroot++; 
	#ifdef DEBUG_traceGridCell
      printf("Lr= %f nr= %f \n", L, cyl_norm[0]);
	#endif
//      goto lab1;

      /*  2)  intersection with planes z_k=const  */
    z_planes:
      if(cyl_norm[2]>0){
	zfc = pM->RootMinX[2] + ((Real)(kcur - ks) +1.)*pG->dx3;
      }
      else if (cyl_norm[2]<0.){
	zfc = pM->RootMinX[2] + ((Real)(kcur - ks))*pG->dx3;	
      }
      else{  /* nz==0 */
	goto phi_planes;	 
      }
      L1 = fabs( (zfc - xyz_pos[2])/cart_norm[2]);

     

      if (L1 <= L){ /*min of Lr or Lz */
       change_indx = 2;
       L=L1;
       nroot++;

	#ifdef DEBUG_traceGridCell
       printf("Lz= %f , nz= %f \n", L1, cyl_norm[2]);
	#endif
      }
   



      /*  3)  intersection with planes phi_j=const */
    phi_planes:

	if( crosPhiConstPlanes == 1){

	  if(cyl_norm[1]>0){
       tfc = pM->RootMinX[1] + ((Real)(jcur - js)+1)*pG->dx2;
      }
      else if(cyl_norm[1]<0.){
       tfc = pM->RootMinX[1] + ((Real)(jcur - js))*pG->dx2;
      }
      else{
      	return;
      }
      sint = sin(tfc);
      cost = cos(tfc);
      d = cart_norm[1]*cost - cart_norm[0]*sint;
      
      if(d != 0.){
      	L1  = fabs((xyz_pos[0]*sint -xyz_pos[1]*cost)/d);
      	if (L1 < L){ /*min of Lr or Lz */
      	  change_indx = 1;
      	  L=L1;
          nroot++;	  
	#ifdef DEBUG_traceGridCell
	  printf("Lt= %f \n", L1);
	#endif
      	}
      }
	}

//lab1:
	  res[0] = L;

      xyz_pos[0] += cart_norm[0]*L; 
      xyz_pos[1] += cart_norm[1]*L;
      xyz_pos[2] += cart_norm[2]*L;

   
      //      if (ijk_cur[change_indx] < ijk_bnd[change_indx][INDS]) ijk_cur[change_indx] = 
      ijk_cur[change_indx] += (int)copysign(1, cyl_norm[change_indx]);
      
      if(change_indx==1){
	if (ijk_cur[1] < js) ijk_cur[1]=je; /* periodic */
	if (ijk_cur[1] > je) ijk_cur[1]=js;
      }

      if(change_indx==0){
      	  if (ijk_cur[0] < is) ijk_cur[1]=is;
      	  if (ijk_cur[0] > ie) ijk_cur[1]=je;
      }
	
      if(change_indx==2){
      	  if (ijk_cur[2] < ks) ijk_cur[2]=ks;
      	  if (ijk_cur[2] > ke) ijk_cur[2]=ke;
      }
	

      
	
      
	#ifdef DEBUG_traceGridCell
      printf(" === end  traceCell === \n");
	#endif
} //end  traceCell



//#define DEBUG_traceGridCell
void traceGridCell(GridS *pG, Real *res, int *ijk_cur, Real *xyz_pos,
		   Real* rtz_pos, const Real *cart_norm, const Real *cyl_norm,
		   short* nroot) {
  /* ijk_cur is the 3d index of the current cell */
  /*  xpos is the (x,y,z) exact cart. position, should be located at one of the cell's boundaries; */
  /*  x1x2x3 is the 3d - r,t,z -center coordinates of the cell to which the 
      above boundaries belong */
  // traces a cingle cell; cnorm is in Cart. coordinates;

  int change_indx, icur, jcur, kcur, crosPhiConstPlanes=0;

  Real a,b,c, rfc, zfc, d,tfc,sint,cost,
    L=HUGE_NUMBER, L1=HUGE_NUMBER, L2=HUGE_NUMBER,
     xcur, ycur, zcur, one_over_a;
     int is,ie,js,je,ks,ke;
     *nroot=0;
     
     is = pG->is;
     ie = pG->ie;
     js = pG->js;
     je = pG->je;
     ks = pG->ks;
     ke = pG->ke;
  
     icur= ijk_cur[0]; /* --- icur etc. are on cyl. grid ----*/
     jcur= ijk_cur[1];
     kcur= ijk_cur[2];

     xcur = xyz_pos[0];   /*  are on Cart. grid */
     ycur=  xyz_pos[1];
     zcur=  xyz_pos[2];


      /*  1)  intersection with cylinders  */     
     
     if(cyl_norm[0]>0){       
       rfc = pG->MinX[0] + ((Real)(icur - pG->is)+1.)*pG->dx1;
     }
     else if (cyl_norm[0]<0.){
       rfc = pG->MinX[0] + ((Real)(icur - pG->is))*pG->dx1;
     }
       else{ /* nr==0: almost never happens*/

	   goto z_planes;
     }
     change_indx = 0;
       
     a = pow(cart_norm[0],2) + pow(cart_norm[1],2);
     b = 2.*(xyz_pos[0]*cart_norm[0]+xyz_pos[1]*cart_norm[1]); 
     c = pow(xyz_pos[0],2)  + pow(xyz_pos[1],2) - pow(rfc,2);
     d = pow(b,2)-4.*a*c;

     if (d<0){ /* possibly at a grazing angle to a cylinder */

   #ifdef DEBUG_traceGridCell
    	 printf("Discr<0. in traceACell\n");
   #endif
       goto z_planes;
     }

      one_over_a =copysign(1./fmax(fabs(a), tiny), a);
      L1 = fabs(-b+sqrt(d))/2.*one_over_a;
      L2 = fabs(-b-sqrt(d))/2.*one_over_a;
      L = fmin(fabs(L1),fabs(L2));
      nroot++; 
	#ifdef DEBUG_traceGridCell
      printf("Lr= %f nr= %f \n", L, cyl_norm[0]);
	#endif
//      goto lab1;

      /*  2)  intersection with planes z_k=const  */
    z_planes:
      if(cyl_norm[2]>0){
	zfc = pG->MinX[2] + ((Real)(kcur - pG->ks) +1.)*pG->dx3;
      }
      else if (cyl_norm[2]<0.){
	zfc = pG->MinX[2] + ((Real)(kcur - pG->ks))*pG->dx3;	
      }
      else{  /* nz==0 */
	goto phi_planes;	 
      }
      L1 = fabs( (zfc - xyz_pos[2])/cart_norm[2]);     

      if (L1 <= L){ /*min of Lr or Lz */
       change_indx = 2;
       L=L1;
       nroot++;

       #ifdef DEBUG_traceGridCell
       printf("Lz= %f , nz= %f \n", L1, cyl_norm[2]);
	#endif
      }
   

      /*  3)  intersection with planes phi_j=const */
    phi_planes:

	if( crosPhiConstPlanes == 1){

	  if(cyl_norm[1]>0){
       tfc = pG->MinX[1] + ((Real)(jcur - pG->js)+1)*pG->dx2;
      }
      else if(cyl_norm[1]<0.){
       tfc = pG->MinX[1] + ((Real)(jcur - pG->js))*pG->dx2;
      }
      else{
      	return;
      }
      sint = sin(tfc);
      cost = cos(tfc);
      d = cart_norm[1]*cost - cart_norm[0]*sint;
      
      if(d != 0.){
      	L1  = fabs((xyz_pos[0]*sint -xyz_pos[1]*cost)/d);
      	if (L1 < L){ /*min of Lr or Lz */
      	  change_indx = 1;
      	  L=L1;
          nroot++;	  
	#ifdef DEBUG_traceGridCell
	  printf("Lt= %f \n", L1);
	#endif
      	}
      }
	}

//lab1:
	  res[0] = L;

      xyz_pos[0] += cart_norm[0]*L; 
      xyz_pos[1] += cart_norm[1]*L;
      xyz_pos[2] += cart_norm[2]*L;

      ijk_cur[change_indx] += (int)copysign(1, cyl_norm[change_indx]);
	#ifdef DEBUG_traceGridCell
      printf(" === end  traceCell === \n");
	#endif
} //end  traceCell

void coordCylToCart(Real *xyz, const Real *rtz,
			   const Real cost, const Real sint ){ 
    xyz[0] = rtz[0]*cost;
    xyz[1] = rtz[0]*sint;
    xyz[2] = rtz[2];
}

void cartVectorToCylVector(Real *cyl, const Real *cart,
			   const Real cost, const Real sint ){
      cyl[0] = cart[0]*cost + cart[1]*sint;
      cyl[1] = -cart[0]*sint + cart[1]*cost;
      cyl[2] = cart[2];
}

void vectBToA(Real* v, const Real* A, const Real*B){
    v[0]= A[0]-B[0];
    v[1]= A[1]-B[1];
    v[2]= A[2]-B[2];      
}

Real absValueVector(const Real* A){
  return( (Real)(sqrt(pow(A[0], 2)+pow(A[1], 2)+pow(A[2], 2))) );
}

void normVectBToA(Real* v, const Real* A, const Real*B, Real* tmp){
    v[0]= A[0]-B[0];
    v[1]= A[1]-B[1];
    v[2]= A[2]-B[2];
    *tmp = absValueVector(v);
    v[0] /= *tmp;
    v[1] /= *tmp;
    v[2] /= *tmp;
}




 void cc_posGlobFromGlobIndx(const MeshS *pM, const GridS *pG, const int iglob,
			     const int jglob,const int kglob, Real *px1, Real *px2, Real *px3)
//returns cell-centered x1,x2,x3 on global mesh, needed for XRAYs; assuming is=0, iterating on global grid
{ 
  *px1 = pM->RootMinX[0] + (Real)(iglob + 0.5)*pG->dx1; 
  *px2 = pM->RootMinX[1] + (Real)(jglob + 0.5)*pG->dx2;
  *px3 = pM->RootMinX[2] + (Real)(kglob + 0.5)*pG->dx3;
  return;
}
 
void cc_posGlob(const MeshS *pM, const GridS *pG, const int iloc, const int jloc,const int kloc,
	    Real *px1, Real *px2, Real *px3)
//returns cell-centered x1,x2,x3 on global mesh, needed for XRAYs
{ 
  *px1 = pM->RootMinX[0] + ((Real)(iloc - pG->is + pG->Disp[0]) + 0.5)*pG->dx1; 
  *px2 = pM->RootMinX[1] + ((Real)(jloc  - pG->js + pG->Disp[1]) + 0.5)*pG->dx2;
  *px3 = pM->RootMinX[2] + ((Real)(kloc - pG->ks + pG->Disp[2]) + 0.5)*pG->dx3;
  return;
}

void ijkLocToGlob(const GridS *pG, const int iloc, const int jloc, const int kloc,
		  int *iglob, int *jglob, int *kglob)
{
  *iglob = iloc - pG->is + pG->Disp[0];  
  *jglob = jloc - pG->js + pG->Disp[1];  
  *kglob = kloc - pG->ks + pG->Disp[2];
  /* printf("%d %d \n%", iloc - pG->is + pG->Disp[0], *iglob); */   
  return;
}

void ijkGlobToLoc(const GridS *pG, const int iglob, const int jglob, const int kglob, int *iloc, int *jloc, int *kloc)
{
  *iloc = iglob + pG->is -  pG->Disp[0];   
  *jloc =  jglob + pG->js -  pG->Disp[1];   
  *kloc = kglob + pG->ks - pG->Disp[2];
 
  return;
}


#ifdef MPI_PARALLEL

 void initGlobComBuffer(const MeshS *pM, const GridS *pG )
{
  
  MPI_Status status;
  MPI_Request request;
  
  int  my_id, ierr,  im, jm, km, size;
  int is,ie,js,je,ks,ke;
  int ext_id,check_id;
  int BufBndr1[6];

  static  int **BufBndrSendRecv;
   
  MPI_Comm_size(MPI_COMM_WORLD, &NumDomsInGlob);
  
   is = pG->is; ie = pG->ie;
   js = pG->js; je = pG->je;
   ks = pG->ks; ke = pG->ke;
    
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   /* printf("NumDomsInGlob, .... my_id, %d %d \n", NumDomsInGlob, my_id ); */
   
   if((recv_rq = (MPI_Request*) calloc_1d_array(NumDomsInGlob,sizeof(MPI_Request))) == NULL)
    ath_error("[problem]: Failed to allocate recv MPI_Request array\n");

   if((send_rq = (MPI_Request*) calloc_1d_array(NumDomsInGlob,sizeof(MPI_Request))) == NULL)
    ath_error("[problem]: Failed to allocate send MPI_Request array\n");

   if((BufBndr = (int**)calloc_2d_array(NumDomsInGlob, 6, sizeof(int) )) == NULL)
      ath_error("[initGlobBuffer]: Failed to allocate BufBndr buffer\n");

    if((BufBndrSendRecv = (int**)calloc_2d_array(NumDomsInGlob, 6, sizeof(int) )) == NULL)
      ath_error("[initGlobBuffer]: Failed to allocate BufBndrSendRecv buffer\n");
 
   
   MPI_Reduce(&ie, &im, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);   
   MPI_Reduce(&je, &jm, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&ke, &km, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);


   /* calculate index position of the local patch in global mesh   */
   /* is -> BufBndr1[0]..ks -> BufBndr1[2]; ie->BufBndr1[2].. */

   ijkLocToGlob(pG, is, js, ks,  &(BufBndr1[0]), &(BufBndr1[1]), &(BufBndr1[2]));
   ijkLocToGlob(pG, ie, je, ke,  &(BufBndr1[3]), &(BufBndr1[4]), &(BufBndr1[5]));

   /* printf(" after ijkLocToGlob BufBndr1:  %d  %d  %d %d  %d  %d  id = %d \n\n", */
   /* 	  BufBndr1[0], BufBndr1[1], BufBndr1[2], */
   /* 	  BufBndr1[3], BufBndr1[4], BufBndr1[5], */
   /* 	  my_id); */

    /* printf(" NumDomsInGlob = %d \n", NumDomsInGlob); */
        

   
    
   if (my_id == 0) {

     for (int ii = 0; ii<=5; ii++) BufBndr[0][ii] = BufBndr1[ii];
     
     /* printf(" im, jm, km =  %d %d %d\n", im, jm, km); */
       size = im*jm*km;
       BufSize = size;
       ibufe = im;
       jbufe = jm;
       kbufe =km;

       // receive message from any source
       for (int i=1; i<NumDomsInGlob; i++){

	 ierr = MPI_Irecv(&(BufBndr1),6,MPI_INT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&request);

	 ierr = MPI_Wait(&request,&status);

	 ext_id=status.MPI_SOURCE;

	 /* copy info about other grids to loc buffer */
	 for (int ii = 0; ii<=5; ii++) BufBndr[ext_id][ii] = BufBndr1[ii];

	   
	 /* printf(" receiving at 0  BufBndr1: %d  %d  %d %d  %d  %d  id_src = %d \n\n", */
	 /* 	BufBndr1[0], BufBndr1[1], BufBndr1[2], */
	 /* 	BufBndr1[3], BufBndr1[4], BufBndr1[5], */
	 /* 	status.MPI_SOURCE); */
	 /* printf(" from %d %d  \n", i, NumDomsInGlob); */	       

       } //received from all my_id(s)


       /* pack and send full root copy of BufBndr */
       int *pSnd = (int*) &(BufBndrSendRecv[0]) ;

       for (int i=0; i<NumDomsInGlob; i++){
	 for(int ii = 0; ii<=5; ii++){	     
	   *(pSnd++) = BufBndr[i][ii];	
	     }
       }

       for (int dest = 1;  dest < NumDomsInGlob; dest++){
	 ierr = MPI_Isend(&(BufBndrSendRecv[0]), NumDomsInGlob*6, MPI_INT, dest, 0,
			  MPI_COMM_WORLD, &request);       
	 /* check non-blocking send has completed. */
	 ierr = MPI_Wait(&(request), &status);       
       }
       
       /* printf(" Received all BufBndr at my_id = 0 : \n"); */

       /* for (int ii = 0; ii<NumDomsInGlob; ii++){ */
       /* 	 printf(" id %d \n", ii); */
       /* 	 printf(" %d  %d  %d %d  %d  %d  \n", */
       /* 		BufBndr[ii][0], BufBndr[ii][1], BufBndr[ii][2], BufBndr[ii][3], BufBndr[ii][4], BufBndr[ii][5]); */
       /* } */
       
       /* MPI_Barrier(MPI_COMM_WORLD); */
       
  
       /* MPI_Bcast(&(BufBndr[0]), 6 * NumDomsInGlob, MPI_INT, 0, MPI_COMM_WORLD); */

       /* printf(" igs, jgs, kgs,  id, %d  %d  %d %d  %d  %d  %d \n", */
       /* 	 	BufBndr[ext_id][0], BufBndr[ext_id][1], BufBndr[ext_id][2], */
       /* 	 	BufBndr[ext_id][3], BufBndr[ext_id][4], BufBndr[ext_id][5], */
       /* 	 	status.MPI_SOURCE); */

   }
   
   else { /* my_id is not 0 */

     ext_id = my_id;

     /* printf("sending my_id from %d to 0\n", ext_id); */
     /* printf(" sending igs, jgs, kgs,  id, %d  %d  %d %d  %d  %d  %d \n", */
     /* 	 	BufBndr1[0], BufBndr1[1], BufBndr1[2], */
     /* 	 	BufBndr1[3], BufBndr1[4], BufBndr1[5], */
     /* 	 	status.MPI_SOURCE); */

     for (int ii = 1; ii<=5; ii++) BufBndr[ext_id][ii] = BufBndr1[ii];
	  
     ierr = MPI_Isend(&(BufBndr1), 6, MPI_INT, 0, 0, MPI_COMM_WORLD, &request);          
     ierr = MPI_Wait(&request, MPI_STATUS_IGNORE);


     
     int *pRcv = (int*) &(BufBndrSendRecv[0]) ;     
     /* receiving full copy of BufBndr from root */
     ierr = MPI_Irecv(&(BufBndrSendRecv[0]), NumDomsInGlob*6, MPI_INT, 0, 0,
		      MPI_COMM_WORLD,  &request);
     /* wait on non-blocking receive and unpack data */
     ierr = MPI_Wait(&request, &status);      

     for (int i=0; i<NumDomsInGlob; i++){
	 for(int ii = 0; ii<=5; ii++){	     
	   BufBndr[i][ii] = *(pRcv++);	
	     }
       }

      /* printf(" test  %d  %d %d  %d  %d  %d id = %d \n\n", */
      /* 	  BufBndr[my_id][0], BufBndr[my_id][1], BufBndr[my_id][2], */
      /* 	  BufBndr[my_id][3], BufBndr[my_id][4], BufBndr[my_id][5], */
      /* 	  my_id); */

     
     
   } /* my_id split end */

   /* broadcasting to all about global buffer */
   MPI_Bcast(&BufSize, 1, MPI_INT,   0, MPI_COMM_WORLD);
   MPI_Bcast(&ibufe,   1, MPI_INT,   0, MPI_COMM_WORLD);
   MPI_Bcast(&jbufe,   1, MPI_INT,   0, MPI_COMM_WORLD);
   MPI_Bcast(&kbufe,   1, MPI_INT,   0, MPI_COMM_WORLD);
   /* printf(" ibufe, jbufe, kbufe, BufSize,  id,  %d  %d  %d  %d %d \n\n", ibufe, jbufe, kbufe, */
   /* 	      BufSize, my_id); */

   /* for(int ii=0; ii < NumDomsInGlob; ii++){ */
   /* 	 printf(" ig, jg, kg,  id, %d  %d  %d %d  %d  %d my_id= %d \n\n", */
   /* 	 	BufBndr[ii][0], BufBndr[ii][1], BufBndr[ii][2], */
   /* 	 	BufBndr[ii][3], BufBndr[ii][4], BufBndr[ii][5], */
   /* 	 	ii); */
   /* } */

   /* big 1d bufer for sending yglob */
   BufSizeGlobArr = pM->Nx[2]* pM->Nx[1]* pM->Nx[0];

 
   /* printf( "BufSizeGlobArr = %d %d \n\n " , BufSizeGlobArr, my_id); */


   if((send_buf = (double**)calloc_1d_array(BufSize,sizeof(double))) == NULL)
      ath_error("[initGlobBuffer]: Failed to allocate send buffer\n");

   if((recv_buf = (double**)calloc_1d_array(BufSize,sizeof(double))) == NULL)
      ath_error("[initGlobBuffer]: Failed to allocate recv buffer\n");
    

   if((send_buf_big = (double**)calloc_1d_array(BufSizeGlobArr,sizeof(double))) == NULL)
      ath_error("[initGlobBuffer]: Failed to allocate send buffer\n");

   if((recv_buf_big = (double**)calloc_1d_array(BufSizeGlobArr,sizeof(double))) == NULL)
      ath_error("[initGlobBuffer]: Failed to allocate recv buffer\n");


   /* printf("   buffer allocation  done at id = %d .... %d \n", my_id, BufSizeGlobArr); */
   return;
}

//#define dbgSyncGridGlob
void SyncGridGlob(MeshS *pM, DomainS *pDomain, GridS *pG, int W2Do)
  /* syncs grid patches to global var */
{
    
  MPI_Status status;
  MPI_Request request;

  int my_id, ext_id, dest, i,j,k;
  int iloc, jloc, kloc;

  /* printf( "from SyncGridGlob %d \n", NumDomsInGlob); */

   
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   
  int ierr;
  
  /* printf("SyncGridGlob::,  my_id = %d,  \n",(int)my_id ); */

 
  if(my_id == 0) {

    // receive message from any source
    for (i=1; i < NumDomsInGlob; i++){

      /* int  mpi1=1; */
      /* while(mpi1 == 1 ) */
      
      ierr = MPI_Irecv(&(recv_buf[0]), BufSize, MPI_DOUBLE, MPI_ANY_SOURCE, 0,
		       MPI_COMM_WORLD,  &request);

      /* wait on non-blocking receive and unpack data */
      ierr = MPI_Wait(&request, &status);
      ext_id=status.MPI_SOURCE;
      
      unPackAndFetchToGlobGrid(pM, pG, &ext_id,W2Do);

#ifdef dbgSyncGridGlob
      printf("unPackAndFetchToGlobGrid done ------- NumDomsInGlob = %d %d \n", NumDomsInGlob, i);
      printf("received yglob from  .. id = %d \n\n", ext_id);      
      aplot(pM, 0,0,0, pM->Nx[0]-1, pM->Nx[1]-1, pM->Nx[2]-1, "ro");
#endif      
    }   

     /* is=0, js=1, ks=2; ie=3, je=4, ke=5 */
     for (k = BufBndr[my_id][2]; k <= BufBndr[my_id][5]; k++) {     
      for (j = BufBndr[my_id][1]; j<=BufBndr[my_id][4]; j++) {
	for (i =BufBndr[my_id][0]; i<=BufBndr[my_id][3]; i++) {
	  iloc = i + pG->is - pG->Disp[0];
	  jloc = j + pG->js - pG->Disp[1];
	  kloc = k + pG->ks - pG->Disp[2];  

	  if (W2Do == ID_DEN){
	    pG->yglob[k][j][i].ro  = pG->U[kloc][jloc][iloc].d;
	  }
	  /* pG->yglob.tau by this time must be already calculated by the local func,
	   so no need to update the root copy*/
	  	  
	}
      }
      /* printf(" %f id = %d \n ", pG->yglob[k][j][i].ro, my_id); */
     }
     // printf("copy  BufBndr done \n\n");
    
 
     /* plot from id0 after sync */
     /* aplot(pM, 0,0,0, pM->Nx[0]-1, pM->Nx[1]-1, pM->Nx[2]-1, 'ro'); */

  
     /* sync yglob from root with all */     
     for (dest = 1;  dest < NumDomsInGlob; dest++){

       packGlobBufForGlobSync(pM, pG, &my_id, W2Do);
       //printf(" done! \n\n" );

       
       ierr = MPI_Isend(&(send_buf_big[0]), BufSizeGlobArr, MPI_DOUBLE, dest, 0,
			MPI_COMM_WORLD, &request);
       /* check non-blocking send has completed. */
       ierr = MPI_Wait(&(request), &status);

       //printf("\n sending from root to id = %d \n\n", dest);
      
     }
     
     /* aplot(pM , BufBndr[my_id][0],BufBndr[my_id][1], BufBndr[my_id][2], */
     /*  	    BufBndr[my_id][3], BufBndr[my_id][4],BufBndr[my_id][5]); */         
     /* aplot(pM,  pG->is,  pG->js,  pG->ks,  pG->ie,  pG->je,  pG->ke); */     
     /* aplot(pM, 0,0,0, pM->Nx[0]-1, pM->Nx[1]-1, pM->Nx[2]-1); */
  }
  else {  /* my_id is not 0 */   
   
    /* printf("BufSize = %d \n", BufSize);  */
    packGridForGlob(pM, pG, &my_id, W2Do);           
    ierr = MPI_Isend(&(send_buf[0]), BufSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request);    
    /* check non-blocking send has completed. */
    ierr = MPI_Wait(&(request), &status); 

    // printf("sending yglob from  .. id = %d \n\n", my_id);
 
     /* receive yglob from id=0 */
     ierr = MPI_Irecv(&(recv_buf_big[0]), BufSizeGlobArr, MPI_DOUBLE, 0, 0,
		      MPI_COMM_WORLD,  &request);

      /* wait on non-blocking receive and unpack data */
      ierr = MPI_Wait(&request, &status);      

      unPackGlobBufForGlobSync(pM, pG, &my_id, W2Do);

      /* printf("\n\n plotting yglob from %", my_id); */
      /* aplot(pM, 0,0,0, pM->Nx[0]-1, pM->Nx[1]-1, pM->Nx[2]-1, "ro"); */
  }
       
  return;

}



static void packGridForGlob(MeshS *pM, GridS *pG, int* myid, int W2Do)
{
  int is = pG->is, ie = pG->ie;
  int js = pG->js, je = pG->je;
  int ks = pG->ks, ke = pG->ke;
  int i,j,k,ig,jg,kg;

  Real *pSnd;
  pSnd = (Real*)&(send_buf[0]);
  
  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
	
 	if (W2Do == ID_DEN){
	  *(pSnd++) = pG->U[k][j][i].d;
	}
	else {
	  ijkLocToGlob(pG, i,j,k, &ig, &jg, &kg);

	  *(pSnd++) =  pG->yglob[kg][jg][ig].tau;

	}
	
      }
    }
  }
  
  return;
}

static void unPackAndFetchToGlobGrid(MeshS *pM, GridS *pG, int *ext_id, int W2Do)
{
  /* unpacks part of remote patch of glob grid and fetches it to the local copy of the glob grid */
  int is = BufBndr[*ext_id][0], ie = BufBndr[*ext_id][3];
  int js = BufBndr[*ext_id][1], je = BufBndr[*ext_id][4];
  int ks = BufBndr[*ext_id][2], ke = BufBndr[*ext_id][5];
  
  int i,j,k;

  double *pRcv;
  pRcv = (double*)&(recv_buf[0]);

  /* printf("unPackGridForGlob %d %d %d %d %d %d id= %d \n\n", is,js, ks,ie, ie,ke,*ext_id); */
  
    for (k=ks; k<=ke; k++) {
      for (j=js; j<=je; j++) {
	for (i=is; i<=ie; i++) {
	  if (W2Do == ID_DEN){
	    pG->yglob[k][j][i].ro  = *(pRcv++);
	  }
	  else {	    
	    pG->yglob[k][j][i].tau = *(pRcv++);
	  }
	 	 	  
	}
      }
    }
    
}


static void packGlobBufForGlobSync(MeshS *pM, GridS *pG, int* myid, int W2Do)
{
  /* packs a local patch of loc copy of glob grid */
   /* sync all of the glob grid */
  int is = 0, ie =  pM->Nx[0]-1;
  int js = 0, je =  pM->Nx[1]-1;
  int ks = 0, ke =  pM->Nx[2]-1;
  int i,j,k;

  Real *pSnd;
  pSnd = (Real*)&(send_buf_big[0]);
  
  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
	if (W2Do == ID_DEN){
	  *(pSnd++) = (pG->yglob[k][j][i]).ro;
	}
	else {
	  *(pSnd++) = (pG->yglob[k][j][i]).tau;
	}

      }
    }
    
  }
 
  return;
}

static void unPackGlobBufForGlobSync(MeshS *pM, GridS *pG, int *ext_id, int W2Do)  
{
   /* unpacks all of remote glob grid and fetches it to the local copy of the glob grid */
   /* syncs all of the glob grid */
  int is = 0, ie =  pM->Nx[0]-1;
  int js = 0, je =  pM->Nx[1]-1;
  int ks = 0, ke =  pM->Nx[2]-1;
  int i,j,k;
  double *pRcv;
  pRcv = (double*)&(recv_buf_big[0]);

  //printf("unPackGlobBufForGlobSync %d %d %d %d %d %d id= %d \n\n", is,js, ks,ie, ie,ke,*ext_id);
  
    for (k=ks; k<=ke; k++) {
      for (j=js; j<=je; j++) {
	for (i=is; i<=ie; i++) {

	if (W2Do == ID_DEN){
	  pG->yglob[k][j][i].ro  = *(pRcv++);
	}
	else {
	  pG->yglob[k][j][i].tau  = *(pRcv++);
	}

	}
      }
    }
}


void freeGlobArrays()
{
  free_1d_array(recv_rq);
  free_1d_array(send_rq);
  free_2d_array(BufBndr);
  free_1d_array(send_buf);
  free_1d_array(send_buf_big);
  free_1d_array(recv_buf);
  free_1d_array(recv_buf_big);    
}

#endif


/* given x, returns containing cell first index.  */
int celli_Glob(const MeshS *pM, const Real x, const Real dx1_1, int *i, Real *a, const Real is)
{
  *a = (x -  pM->RootMinX[0]) * dx1_1 + is;
  *i = (int)(*a);
  if (((*a)-(*i)) < 0.5) return 0;	/* in the left half of the cell*/
  else return 1;			/* in the right half of the cell*/
}

int cellj_Glob(const MeshS *pM, const Real y, const Real dx2_1, int *j, Real *b,const Real js)
{
  *b = (y - pM->RootMinX[1]) * dx2_1 + js;
  *j = (int)(*b);
  if (((*b)-(*j)) < 0.5) return 0;	/* in the left half of the cell*/
  else return 1;			/* in the right half of the cell*/
}

int cellk_Glob(const MeshS *pM, const Real z, const Real dx3_1, int *k, Real *c, const Real ks)
{
  *c = (z - pM->RootMinX[2]) * dx3_1 + ks;
  *k = (int)(*c);
  if (((*c)-(*k)) < 0.5) return 0;	/* in the left half of the cell*/
  else return 1;			/* in the right half of the cell*/
}






