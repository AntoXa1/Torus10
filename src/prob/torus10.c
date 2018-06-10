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
#include <stdbool.h>
#include <string.h>
#include <time.h>

#include "defs.h"
#include "athena.h"
#include "globals.h"
#include "prototypes.h"
#include<limits.h> 
//****************************************
#ifdef MPI_PARALLEL

//#define DBG_MPI_OPT_STACK //mpi "breakpoint" while loops

#endif

#define HAWL

//#define RADIAL_RAY_TEST
//#define SOLOV

//#define tau_sync_small_buf
//#define mpi_opt_depth_test1


//****************************************
/* make all MACHINE=macosxmpi */


static void inX1BoundCond(GridS *pGrid);
static void diode_outflow_ix3(GridS *pGrid);
static void diode_outflow_ox3(GridS *pGrid);

void plot(MeshS *pM, char name[16]);
void aplot(MeshS *pM, int is, int js, int ks, int ie, int je, int ke, char name[16]);


/* void aplot(GridS *pG, int is, int ie, int js, int je, int ks, int ke); */
// void plot(MeshS *pM, char name[16]);
static void calcProblemParameters();
static void printProblemParameters();


float ellf(float phi, float ak);
float rd(float x, float y, float z);
float rf(float x, float y, float z);  


// void Constr_optDepthStack(MeshS *pM, GridS *pG);

void Constr_RayCastingOnGlobGrid(MeshS *pM, GridS *pG, int my_id);
// void optDepthStack(MeshS *pM, GridS *pG);

void findSegmPartitionFromRay(GridS *pG, RayData GridOfRays, int,int, RaySegment* tmpSegGrid, int* NumSeg,
			      int* tmp_mpiIdVec,
			      int NumMPIBlocks, int my_id);

void findSegmPartitionFromRayForBlock(GridS *pG, RayData GridOfRays,
			      int,int, RaySegment* tmpSegGrid, int* NumSeg,
			      int* tmp_mpiIdVec,
			      int NumMPIBlocks, int my_id);




void Constr_SegmentsFromRays(MeshS *pM, GridS *pG, int);

void SearchBufId(int i, int k, int NumMpiBlocks, int* id_res);
  
enum BCBufId CheckCrossBlockBdry(int i, int k, int *my_id);


void testSegments1(GridS *pG, int ip, int kp, RaySegment* tmpSegGrid, int);
void testSegments2(MeshS *pM, GridS *pG, int my_id, int);

void OptDepthAllSegBlock(GridS *pG);
void CalcParallelOdepthData(GridS *pG, int my_id);
  


  
void optDepthStackOnGlobGrid(MeshS *pM, GridS *pG, int my_id);
void ionizParam(const MeshS *pM, GridS *pG);
void optDepthFunctions(GridS *pG);

Real updateEnergyFromXrayHeatCool(const Real E, const Real d,
				  const Real M1,const Real M2, const Real M3,
				  const Real B1c,const Real B2c,const Real B3c,
				  const Real xi, const Real dt,
				  int i, int j, int k);



void xRayHeatCool(const Real dens, const Real Press, const Real xi_in, Real*, Real*,
		  const Real dt);

Real rtsafe_energy_eq(Real, Real, Real, Real, int*);

/* ============================================================================= */
/*                       "gravity" functions                                   */
/* ============================================================================= */
void calcFFT();


/* ============================================================================= */
/*                       ray tracing functions                                   */
/* ============================================================================= */
void testRayTracings( MeshS *pM, GridS *pG);
void traceGridCell(GridS *pG, Real *res, int ijk_cur[3], Real xpos[3], Real* x1x2x3, const Real
		   cartnorm[3], const Real cylnorm[3], int*);
void traceGridCellOnGlobGrid(MeshS *pM,GridS *pG, Real *res, int *ijk_cur, Real *xyz_pos,
			     Real* rtz_pos, const Real *cart_norm, const Real *cyl_norm,			    
			     int* nroot);
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

int celli_Glob(const MeshS *pM, const Real x, const Real dx1_1, int *i, Real *a, const int is);
int cellj_Glob(const MeshS *pM, const Real y, const Real dx2_1, int *j, Real *b, const int js);
int cellk_Glob(const MeshS *pM, const Real z, const Real dx3_1, int *k, Real *c, const int ks);

 /* subroutines related to Ray object on global grid */
void initGlobComBuffer(const MeshS *pM, const GridS *pG );
void AllocateSendRecvGlobBuffers(const MeshS *pM, const GridS *pG);
/* void SyncGridGlob(MeshS *pM, DomainS *pDomain, GridS *pG, int W2Do); */

#ifdef use_glob_vars
void SyncGridGlob(MeshS *pM, GridS *pG, int W2Do);
static void packGridForGlob(MeshS *pM, GridS *pG, int* my_id, int W2Do);
static void packGlobBufForGlobSync(MeshS *pM, GridS *pG, int* myid, int W2Do);
static void unPackAndFetchToGlobGrid(MeshS *pM, GridS *pG, int* ext_id, int W2Do);
static void unPackGlobBufForGlobSync(MeshS *pM, GridS *pG, int* ext_id, int W2Do);
#endif

void freeGlobArrays();

/* int  mpi1=1; */
int NumSegBlock_glb=0;
static int my_id_glob=-1;

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


int ID_DEN = 0;
int ID_TAU = 1;

#ifdef MPI_PARALLEL
/* MPI send and receive buffers */
int  BufSize, BufSizeGlobArr, ibufe, jbufe, kbufe, **BufBndr, NumDomsInGlob; //sizes of global buffer 

#ifdef use_glob_vars 
static double **send_buf = NULL, **recv_buf = NULL,
  **send_buf_big = NULL, **recv_buf_big = NULL;
#endif 

static MPI_Request *recv_rq, *send_rq;

float *send_1d_buf=NULL,  *recv_1d_buf=NULL;

#endif



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



#ifdef MPI_PARALLEL
#ifdef use_glob_vars 
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

	for (ii = 0; ii<(pG->GridOfRays[kp][ip]).len; ii++){

	  i=(pG->GridOfRays[kp][ip]).Ray[ii].i;
	  j = jp;
	  j=(pG->GridOfRays[kp][ip]).Ray[ii].j;	  
	  k=(pG->GridOfRays[kp][ip]).Ray[ii].k;

	  //den = pG->U[kloc][jloc][iloc].d;

	  den= 	pG->yglob[k][j][i].ro;
	  dl = (pG->GridOfRays[kp][ip].Ray[ii]).dl;	  	    	 	  
	  dtau = Rsc*Dsc*KPE* dl*den;	  
          tau += dtau;

	} /* eof ray iter */

	pG->yglob[kp][jp][ip].tau = tau;
        tau = 0;
	
      } //ip
    } //jp
  } //kp

}
#endif
#endif

void Constr_RayCastingOnGlobGrid(MeshS *pM, GridS *pG, int my_id){
 
  // it is assumed that the source is located on the axis of symmetry


  Real x1is, x2js, x3ks, den, ri, tj, zk,x1,x2,x3,
    l,dl,tau=0,dtau=0 ;
   
  int i,j,k,is,ie,js,je,ks,ke,ip,jp,kp,iloc,jloc,kloc,
    knew,ii;

  Real abs_cart_norm, cart_norm[3], cyl_norm[3], xyz_src[3], xyz_pos[3],
    rtz_pos[3],rtz_in[3], xyz_p[3],
    res[1], olpos[3],xyz_cc[3], rtz_cc[3];
  int ijk_cur[3],iter,iter_max,lr,i0,j0,k0,
    NmaxArray=0;
  int nroot;
  Real xyz_in[3], radSrcCyl[3], dist, sint, cost, s2,ir=0;
    
  //    CellOnRayData arrayDataTmp; //indices, ijk and dl
  //    int tmpIntArray1, *tmpIntArray2, *tmpIntArray3;

  CellOnRayData *tmpCellIndexAndDisArray;
  Real *tmpRealArray;
  int ncros;
  int  mpi1=1; 
  
  is = 0;
  ie = pM->Nx[0]-1;

  js = 0;
  je = pM->Nx[1]-1;       

  ks = 0;
  ke = pM->Nx[2]-1;

  /* printf("ke = pM->Nx[2] = %d \n", pM->Nx[2]); */
  
  iter_max= 5*sqrt(pow(ie,2) + pow(je,2) +pow(ke,2));

  //  allocating temporary array for 1D ray from point source
  //  CellIndexAndCoords

  if ((tmpCellIndexAndDisArray=(CellOnRayData*)calloc(iter_max ,
						      sizeof(CellIndexAndCoords)))== NULL) {
    ath_error("[calloc_1d] failed to allocate memory CellIndexAndCoords\n");
  }

  cc_posGlobFromGlobIndx(pM, pG,is,js,ks, &x1is,&x2js,&x3ks);

  //fc_pos(pG,is,js,ks, &x1is,&x2js,&x3ks);


  jp = js;
		
  for (kp = ks; kp <= ke; kp++) { //z
    for (ip = is; ip <= ie; ip++) { //r
      NmaxArray =0;
#ifndef MPI_PARALLEL /* if Not parallel */
      ath_error("[Constr_RayCastingOnGlobGrid] is not working in parallel\n");
      /* pG->yglob[kp][jp][ip].ro = pG->U[kloc][jloc][iloc].d; */
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
	
      lr=celli_Glob(pM, rtz_in[0], 1./pG->dx1, &i0, &ir, is); 

      lr=cellj_Glob(pM,rtz_in[1], 1./pG->dx2, &j0, &ir, js);

      /* lr=cellj_Glob(pM,rtz_in[1], 1./pG->dx2, &j0, &ir, jp); */
	
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

      // zero element
      iter=0;
      (tmpCellIndexAndDisArray[iter]).dl = dl;
      (tmpCellIndexAndDisArray[iter]).i = ijk_cur[0];
      (tmpCellIndexAndDisArray[iter]).j = ijk_cur[1];
      (tmpCellIndexAndDisArray[iter]).k = ijk_cur[2];
      NmaxArray += 1;
      
      for (iter=1; iter<=iter_max; iter++){
	/* test if  already in ip,kp */
	if( (ijk_cur[0]==ip  &&  ijk_cur[2]==kp)) break;
	
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

	(tmpCellIndexAndDisArray[iter]).dl = dl;
	(tmpCellIndexAndDisArray[iter]).i = ijk_cur[0];
	(tmpCellIndexAndDisArray[iter]).j = ijk_cur[1];	  
	(tmpCellIndexAndDisArray[iter]).k = ijk_cur[2];

	l += dl;
       	  
	NmaxArray = iter+1; //To use in allocation

	if (pow(xyz_p[0]-xyz_pos[0],2)+pow(xyz_p[1]-xyz_pos[1],2)+
	    pow(xyz_p[2]-xyz_pos[2],2)>s2){
	    
	  //  tau= 0;
	  //  break;
	}
		 
	/* test if reached to the ip,jp,kp */
	if( (ijk_cur[0]==ip  &&  ijk_cur[2]==kp)) break;

      } //iter
	
      //pG->GridOfRays should be already allocated at init_grid.c
      //{ printf("MPI breakpoint, id=%d \n", my_id); int mpi1= 1;  while(mpi1 == 1);}

      /* printf("%d %d %d \n", NmaxArray, ip, jp); */
      /* { printf("MPI breakpoint2, id=%d \n", my_id); int mpi1= 1;  while(mpi1 == 1);}	      */


      
      (pG->GridOfRays[kp][ip]).Ray =
	(CellOnRayData*)calloc_1d_array(NmaxArray,sizeof(CellOnRayData));
      
      (pG->GridOfRays[kp][ip]).len = NmaxArray;      
      
      for (ii = 0; ii<NmaxArray; ii++){

	(pG->GridOfRays[kp][ip]).Ray[ii].dl= (tmpCellIndexAndDisArray[ii]).dl;
	  
	(pG->GridOfRays[kp][ip]).Ray[ii].i=(tmpCellIndexAndDisArray[ii]).i;	  
	(pG->GridOfRays[kp][ip]).Ray[ii].j=(tmpCellIndexAndDisArray[ii]).j;
	(pG->GridOfRays[kp][ip]).Ray[ii].k=(tmpCellIndexAndDisArray[ii]).k;

	  
	//		   printf("NmaxArray= %d dl \n\n ", NmaxArray);	
      } //over >GridOfRays = tmpCellIndexAndDisArray      				      

      int mpi=1;
      //while((mpi==1) && (my_id==1) && (ip==9));
	
      l=0;
      tau=0;

    } //ip    
  } //kp

      

  free(tmpCellIndexAndDisArray);

  
    /* for (kp=pG->ks; kp<=pG->ke; kp++){      */
    /*    for(ip=pG->is; ip<=pG->ie; ip++{	   */
    /* 	  ijkGlobToLoc(pG, ip, jp, kp, &iloc, &jloc, &kloc);  */
    /*      } */
    /*  } */
   
    }
void Constr_SegmentsFromRays(MeshS *pM, GridS *pG, int my_id){
  /*! breaks the grid of rays, obtained from global grid into individual segments
    belonging to a particular MPI Block = Grid */
  
  int ig,ig_s,ig_e, kg,kg_s,kg_e, jg, je,ii, i,k,NumSeg=0;
  int NumOfSegInBlock,NumMPIBlocks,cur_id,pr_id,mpi1;

  int *tmp_mpiIdVec;        //tmp stores mpi ids of segments
  
  /* enum SegmentType *tmp_SegType;   //tmp stores array of segments type  */

  RaySegment *tmpSegGrid; /* temp array: we don't know how many segs per Block */ 
  /*   temporary arrays are needed because we don't know how 
       many ray segments are crossing a given MPI Block; 1) we allocate enough space; 
       then copy the results to a perm structure of segments and deallocate tmp structures; */
  ig_s = 0;
  ig_e = pM->Nx[0]-1;;
  kg_s = 0;
  kg_e = pM->Nx[2]-1;;
  
  MPI_Comm_size(MPI_COMM_WORLD, &NumMPIBlocks);
  
  int N = (ig_e-ig_s)*(kg_e-kg_s);
  int N2 = 2*N;
 

  /* printf("N22 = N2 * N2, %d %d\n", N22, N); */
  /* printf("int %%d %d %d to %d \n",   */
  /* 	 sizeof(int),INT_MIN,INT_MAX); */
  
   if( (tmp_mpiIdVec = (int*)calloc_1d_array(N2, sizeof(int)))==NULL)    
      ath_error("[calloc_1d] failed to allocate mpiIdVec \n");

   //{ printf("MPI breakpoint, id=%d \n", my_id); int mpi1= 1;  while(mpi1 == 1);}
 
    if( (tmpSegGrid = (RaySegment*)calloc_1d_array(N2, sizeof(RaySegment)))==NULL)
      ath_error("[calloc_1d] failed to allocate tmpRaySegGridPerBlock \n");

  /* (pG->RaySegGridPerBlock) = */
  /*     (RaySegment*)calloc_1d_array(NumSegBlock_glb,sizeof(RaySegment)); */
  
  jg = 0; /* rays must be in phi_j =const planes */
  NumSegBlock_glb = 0;  
  
  for (kg=kg_s; kg<=kg_e; kg++) {   // z
    for (ig=ig_s; ig<=ig_e; ig++) { //R
      
      for(int m=0; m<N2; m++) tmp_mpiIdVec[m]=-1; //i.e. default is no-Id       
      NumSeg=0;           

      findSegmPartitionFromRayForBlock(pG, pG->GridOfRays[kg][ig], ig,kg,
			       tmpSegGrid, &NumSeg, tmp_mpiIdVec,
			       NumMPIBlocks, my_id);

      NumSegBlock_glb += NumSeg; //increase only if my_block is crossed 

    } //ig
  } //kg
  /* partition of the GridOfRays structure array into segments done */


#ifdef save_memory /* free some memory */
  {
    int is = 0;
    int ie = pM->Nx[0]-1;
    int js = 0;
    int je = pM->Nx[1]-1;       
    int ks = 0;
    int ke = pM->Nx[2]-1;
    for (int kp = ks; kp<=ke; kp++) { //z
      for (int ip = is; ip<=ie; ip++) { //r
	free_1d_array((pG->GridOfRays[kp][ip]).Ray);
      }
    }

    free_2d_array(pG->GridOfRays);
  } 
#endif
 
  
   /* mpi1=1; */
   /* while(mpi1==1); */

 
  
  { //#2 copy temporary rays to perm structure   
    (pG->RaySegGridPerBlock) =
      (RaySegment*)calloc_1d_array(NumSegBlock_glb,sizeof(RaySegment));
    
    for(int l = 0; l <  NumSegBlock_glb; l++){
      
      if(((pG->RaySegGridPerBlock)[l].data=
	  (SegmentData*)calloc_1d_array(tmpSegGrid[l].data_len,sizeof(SegmentData)))==NULL)	
	ath_error("[calloc_1d] failed to allocate (pG->RaySegGridPerBlock) \n");

     
       
      (pG->RaySegGridPerBlock)[l].data_len=tmpSegGrid[l].data_len;

      (pG->RaySegGridPerBlock)[l].type=tmpSegGrid[l].type;

      /* if(my_id==0) printf("l= %d %d %d \n", l, (pG->RaySegGridPerBlock)[l].type, */
      /* 			  tmpSegGrid[l].type); */
      
      for(int b=0; b<4; b++) pG->RaySegGridPerBlock[l].head_block[b] =
			       tmpSegGrid[l].head_block[b];
                  
      for(int m=0; m < (pG->RaySegGridPerBlock)[l].data_len; m++){

	/* printf("\n (l,m)=(%d %d) \n", l, m); */	
	(pG->RaySegGridPerBlock)[l].data[m].i = tmpSegGrid[l].data[m].i;
	(pG->RaySegGridPerBlock)[l].data[m].k = tmpSegGrid[l].data[m].k;
	(pG->RaySegGridPerBlock)[l].data[m].dl = tmpSegGrid[l].data[m].dl;

	/* allocate tmp MPI connection array */	
	if( pG->RaySegGridPerBlock[l].type == Head){

	  int sz = (pG->RaySegGridPerBlock)[l].MPI_idConnectArray_size=
	    tmpSegGrid[l].MPI_idConnectArray_size;
	  	  
	  if((pG->RaySegGridPerBlock[l].MPI_idConnectArray=
	      (int*)calloc_1d_array(sz, sizeof(int)))==NULL)
	    ath_error("[Constr_SegmentsFromRays] failed to allocate MPI_idConnectArray)\n");

	  //{printf("MPI breakpoint, id=%d \n", my_id); int mpi1= 1;  while(mpi1 == 1);}

	  /* copy MPI connection array to tmp structure */
	  for(int impi=0; impi <= sz-1; impi++)
	    pG->RaySegGridPerBlock[l].MPI_idConnectArray[impi]=tmp_mpiIdVec[impi];	 
	}   

      } //m-loop      

      { //allocate dtau, phi array
	int js =  BufBndr[my_id][1];
	int je = BufBndr[my_id][4] ;
	int phi_sz = je-js+1;

	/* if( pG->RaySegGridPerBlock[l].type == Upstr ){

	  /* printf("alloc: l= %d phi_sz=, %d %d %d %d \n", l, phi_sz, js, je, */
	  /* 	 pG->RaySegGridPerBlock[l].type); */
	  
	  /* tau_phi array exists only for segments which pass through */

	  pG->RaySegGridPerBlock[l].dat_vec_1d_size = phi_sz;

	  if(( pG->RaySegGridPerBlock[l].dat_vec_1d=
	     (float*)calloc_1d_array(phi_sz,sizeof(float)))==NULL)
	  ath_error("[calloc_1d] failed to allocate (pG->RaySegGridPerBlock)[l].d_tau \n");  
	
      }
      
    } // l-loop    
  }// #2

  /* if(my_id==0)  */
  /* for(int l = 0; l <  NumSegBlock_glb; l++) */
  /*  printf("laft= %d %d %d \n", l, (pG->RaySegGridPerBlock)[l].type, */
  /* 			  tmpSegGrid[l].type); */
  /* mpi1=1; */
  /* while(mpi1==1); */

  
   
  { //deallocating large tmp segment structure
    for(int l=0;l<=N2-1;l++){
      //      if(tmpSegGrid[l].type==Head) free(tmp_mpiIdVec);
      free(tmpSegGrid[l].data);
    }
    free(tmp_mpiIdVec);
    free(tmpSegGrid);  //down the road make sure to proper3ly
    printf("\n %d deallocating large tmp segment structure done ..\n",my_id);
  }
 
 
  
  {//constructs mask array       
    int N2 = pM->Nx[0]*pM->Nx[2];
    if(( pG->BlockToRaySegMask=
	 (int**)calloc_2d_array(pM->Nx[2], pM->Nx[0],sizeof(int)))==NULL)
      ath_error("[calloc_2d] failed to allocate pG->RaySegMask \n");

    int ig_s = BufBndr[my_id][0];
    int ig_e = BufBndr[my_id][3];
    int kg_s = BufBndr[my_id][2];
    int kg_e = BufBndr[my_id][5];
    
    for(int kp=kg_s; kp<=kg_e; kp++){
      for(int ip=ig_s; ip<=ig_e; ip++){

    	int l1, dlen, i,k,l;
	
    	for (int l=0; l < NumSegBlock_glb; l++){
	  
   	  dlen = (pG->RaySegGridPerBlock)[l].data_len;
	  
	  i = (pG->RaySegGridPerBlock)[l].data[dlen-1].i;	  
    	  k = (pG->RaySegGridPerBlock)[l].data[dlen-1].k;
	  
	  if ((i==ip) && (k==kp) && (pG->RaySegGridPerBlock[l].type==Head)){	    
	    l1 = l;
	    break;
	  }
	  else{
	    l1=-1;
	  }
    	}

    	if(l1==-1){
	  mpi1=1;
	  while(mpi1==1);

	  printf("(pG->RaySegGridPerBlock)[l].type= %d \n",(pG->RaySegGridPerBlock)[l].type);
	  printf("id:%d, is:%d,ie:%d, ks:%d,ke:%d\n",my_id, ig_s,ig_e,kg_s,kg_e);
	  printf("id:%d (l-1:%d), (dlen:%d),(ip,kp:%d,%d), (i,k: %d,%d) , (l1:%d)\n",
		 my_id, l, dlen, ip,kp, i,k, l1);
    	  ath_error("could not fill pG->BlockToRaySegMask");
    	}
    	else {
    	  pG->BlockToRaySegMask[kp][ip]=l1;
    	}	
      }
    }        
  }//ends mask array

   
  //test
  /* testSegments2(pM, pG, my_id, 2);     */
  //  printf("\n ********** testSegments2 id: %d *********** \n",  my_id);

     
  /* mpi1=1; */
  /* while(mpi1==1); */  
}


void OptDepthAllSegBlock(GridS *pG){  

  /* int mpi1=1; */
  /* while(mpi1==1);  */
  
  int phi_sz =  BufBndr[my_id_glob][4] - BufBndr[my_id_glob][1]+1;
   
  for(int l=0; l < NumSegBlock_glb; l++){

    Real dat=0.;
    Real den = 1;    
    Real tau0 = Rsc*Dsc*KPE;
        
    int sz =  pG->RaySegGridPerBlock[l].dat_vec_1d_size;

    for(int m = 0; m < sz; m++) {

      dat =0;
      for(int n=0; n <(pG->RaySegGridPerBlock)[l].data_len; n++){     

 		
	int i = pG->RaySegGridPerBlock[l].data[n].i;
	int k = pG->RaySegGridPerBlock[l].data[n].k;

	int iloc, jloc, kloc;     
	ijkGlobToLoc(pG, i, m, k, &iloc, &jloc, &kloc);
		
	den = pG->U[kloc][jloc][iloc].d;

	Real dtau = (pG->RaySegGridPerBlock)[l].data[n].dl*den;
	  
	dat += dtau;
	
	
#ifndef mpi_opt_depth_test1	
	/* dat *= den; */
#endif
	
      }

#ifndef mpi_opt_depth_test1


      pG->RaySegGridPerBlock[l].dat_vec_1d[m] = tau0*dat;

      
#else
      pG->RaySegGridPerBlock[l].dat_vec_1d[m] = dat;
      
#endif
      
      /* pG->yglob[k][m][i].tau = pG->RaySegGridPerBlock[l].dat_vec_1d[m]; */
      /* pG->tau_e[kloc][jloc][iloc] = */
      
    }
  } 

  MPI_Barrier(MPI_COMM_WORLD);
  
}

void findSegmPartitionFromRayForBlock(GridS *pG, RayData GridOfRays,int ig,int kg,
			      RaySegment* tmpSegGrid,
			      int* NumSeg,
			      int* tmp_mpiIdVec,
			      int NumMPIBlocks, int my_id){
  int ii,i,j,k,ip,cur_id,prev_id,fut_id,orig_id,is_s,is_e;

  *NumSeg = 0;
   
  cur_id = -1;
  prev_id = -1;
  fut_id = -1;
  orig_id = -1;
  
  int n_seg=0;
  /* NumSeg is number of segments crossing Block; n_seg is number of segments in a ray */
  /* segment is a part of a ray that belongs to an MPI block */

  int mpi1=1;  

   /* while((mpi1==1) && (my_id==0) && (ig==1)); */
  
  SearchBufId(ig, kg, NumMPIBlocks, &orig_id);
  
  for (ii = 0; ii< GridOfRays.len; ii++){
  
    SearchBufId(GridOfRays.Ray[ii].i, GridOfRays.Ray[ii].k, NumMPIBlocks, &cur_id);

    /* look ahead */
    if (ii< GridOfRays.len-1){
      SearchBufId(GridOfRays.Ray[ii+1].i, GridOfRays.Ray[ii+1].k, NumMPIBlocks, &fut_id);
    }

    if(prev_id != cur_id){

      tmp_mpiIdVec[n_seg]=cur_id;    
      n_seg+=1;
      
      if (cur_id==my_id){ //first cell of my_id block      
	is_s=ii;	 
	*NumSeg += 1;
      }	
      prev_id = cur_id;
    }

    if((cur_id==my_id) && (fut_id!=cur_id)||(ii==GridOfRays.len-1)){
      //last cell of my_id block
      is_e = ii;
      break;
    }    
    }//eof iteration over a ray(ig,kg)

  if (*NumSeg==1){/* this condition simply means that there IS as segment here since some */
    /* Rays may not cross this Block(my_id) */
    
    int seg_i = NumSegBlock_glb; //i.e. + 1 -1

    tmpSegGrid[seg_i].head_block[0] = orig_id;
    tmpSegGrid[seg_i].head_block[1] = ig;
    tmpSegGrid[seg_i].head_block[3] = kg;

    /* printf("my_id=%d, orig_id=%d \n", my_id, orig_id); */
    
    if (orig_id==my_id){
      
      tmpSegGrid[seg_i].type = Head;
      tmpSegGrid[seg_i].MPI_idConnectArray_size = n_seg;

      int sz = tmpSegGrid[seg_i].MPI_idConnectArray_size;
      
      /* printf("alloc id,ig,kg, seg_i,sz %d %d %d \n", my_id, ig, kg, seg_i, sz); */
      /* allocate tmp MPI connection array */     
      
      if( (tmpSegGrid[seg_i].MPI_idConnectArray=
	   (int*)calloc_1d_array(sz,sizeof(int)))==NULL){
        ath_error("failed to allocate tmpSegGrid[seg_i].MPI_idConnectArray\n");
      }
      
    }
    else{
      tmpSegGrid[seg_i].type = Upstr;
      tmpSegGrid[seg_i].MPI_idConnectArray_size = 0;
    }

    tmpSegGrid[seg_i].data_len = is_e-is_s+1;
    
    if( (tmpSegGrid[seg_i].data=
	 (SegmentData*)calloc_1d_array(tmpSegGrid[seg_i].data_len,
				       sizeof(SegmentData)))==NULL)
      ath_error("[calloc_1d] failed to allocate findSegmPartitionFromRayForGrid, 1 \n");

    for(ip = is_s; ip<= is_e; ip++){ //is_s and is_e are in the Ray structure
      int l = ip - is_s;
     
      tmpSegGrid[seg_i].data[l].i = GridOfRays.Ray[ip].i;
      tmpSegGrid[seg_i].data[l].k = GridOfRays.Ray[ip].k;
      tmpSegGrid[seg_i].data[l].dl = GridOfRays.Ray[ip].dl;
    }    
  }
        
  }


void SendSegmentData(int Send1Recv0, int dest_id, float* dat_vec_1d, int phi_sz, int* head_block){
   /* not working correctly */
  /* call with:  SendSegmentData(1, dest_id, pG->RaySegGridPerBlock[l].dat_vec_1d, phi_sz, */
  /* 	 		 pG->RaySegGridPerBlock[l].head_block */
  /* 	 		 ); */

     
     //Create MPI datatype
      MPI_Request request;
      int tag_send = 1;
      
      const int nblocks = 2;

      int blocklengths[2] = {4, phi_sz};
      
      MPI_Datatype types[2] = {MPI_INT, MPI_FLOAT};      
      MPI_Datatype mpi_seg_data;      
      MPI_Aint offsets[3];

      offsets[0] = offsetof(SegmSendRecv, id_ijk);
      offsets[1] = offsetof(SegmSendRecv, dat_vec);

/* int mpi1=1; */
/*    while(mpi1==1); */
      
      SegmSendRecv SegSendBuf, SegRecvBuf;
      
      
      if( (SegSendBuf.dat_vec = (float*)calloc_1d_array(phi_sz, sizeof(float)))==NULL)
        ath_error("[calloc_1d] failed to allocate dat_vec \n");

      if( (SegRecvBuf.dat_vec = (float*)calloc_1d_array(phi_sz, sizeof(float)))==NULL)
        ath_error("[calloc_1d] failed to allocate dat_vec \n");

      MPI_Type_create_struct(nblocks, blocklengths, offsets, types, &mpi_seg_data);

      MPI_Type_commit(&mpi_seg_data);
      
      
      MPI_Status status;
      
      switch (Send1Recv0){
      case (1):
	for(int i=0; i<4; i++) SegSendBuf.id_ijk[i] = head_block[i];
	for(int i=0; i<phi_sz; i++) SegSendBuf.dat_vec[i]=dat_vec_1d[i]=my_id_glob+100;
	  
	/* printf("%d %d %d \n", head_block[1],head_block[2],head_block[3]); */

	int ierr = MPI_Isend(&SegSendBuf, 1, mpi_seg_data, dest_id, tag_send,
			     MPI_COMM_WORLD, &request);

	/* printf("send %d to %d \n",my_id_glob, dest_id); */
	/* for(int i=0; i<phi_sz; i++) printf("%f \n", SegSendBuf.dat_vec[i]); */

	break;
      case (0):

	MPI_Recv(&SegRecvBuf, 1, mpi_seg_data, MPI_ANY_SOURCE, tag_send, MPI_COMM_WORLD, &status);

	/* printf("at id= %d, from: %d ijk: %d,%d,%d\n", my_id_glob, status.MPI_SOURCE);	 */

	//for(int i=0; i<phi_sz; i++) printf("%f \n", SegRecvBuf.dat_vec[i]);
	
	break;

      default:
	ath_error("unknown error in SendSegmentData");
      }
      MPI_Type_free(&mpi_seg_data);
      free(SegSendBuf.dat_vec);      
      free(SegRecvBuf.dat_vec);
  }

void testSegments2(MeshS *pM, GridS *pG, int my_id, int TestType){

  int ig_s = BufBndr[my_id][0];
  int ig_e = BufBndr[my_id][3];
  int kg_s = BufBndr[my_id][2];
  int kg_e = BufBndr[my_id][5];
  int js =  BufBndr[my_id][1];    
  int je = BufBndr[my_id][4] ;

  int phi_sz = je-js+1;
  
  my_id_glob=my_id;

  int nmsg_snd=0, nmsg_recv=0;
  int tag_send = 10;
  
  int n_tot_ms_snd = 0; /* total msg to send this.Grid*/
  int n_tot_ms_rcv = 0; /* total msg to recv */

  float err;
	 
  MPI_Request request;
  MPI_Status status;

  bool receive, send;

  int NumMPIBlocks;
  MPI_Comm_size(MPI_COMM_WORLD, &NumMPIBlocks);
  
  
  int buf_size;
  int dlen;
  int num_extra = 4;
    
  buf_size = phi_sz + num_extra;
 
  float *recv_1d_buf=NULL, *send_1d_buf=NULL;
    
  if( (recv_1d_buf = (float*)calloc_1d_array(buf_size, sizeof(float)))==NULL)
    ath_error("[calloc_1d] failed to allocate send_1d_buf \n");

  int *segm_mask_send; //stores refs to segments to be sent
  int l_snd = 0; //send-segment index 
  int l_rcv = 0; //recvd-segment index 
    
  switch (TestType){
  case 1:   
    //printf("id: %d, (l,%d), %d, %d, \n", my_id, l, kp, ip);
    break;

  case 2:

    //plot(pM, "tau");
            
    //OptDepthAllSegBlock(pG);

    
 
    for(int l=0; l <= NumSegBlock_glb-1; l++){
      if (pG->RaySegGridPerBlock[l].type == Upstr) n_tot_ms_snd +=1;
    }
    /* printf("-----------------------\n"); */
    /* printf("id %d; n_tot_ms_snd= %d \n", my_id, n_tot_ms_snd); */

    /* /\* prepare the big buffer to send *\/ */
    /* if( (send_2d_buf = (float*)calloc_2d_array(n_tot_ms_snd, buf_size, */
    /* 					       sizeof(float)))==NULL) */
    /* ath_error("[calloc_1d] failed to allocate send_1d_buf \n"); */

     /* int mpi1=1; */
     /* while(mpi1==1); */

#ifdef tau_sync_small_buf
    
    if( (send_1d_buf = (float*)calloc_1d_array(buf_size, sizeof(float)))==NULL)
    ath_error("[calloc_1d] failed to allocate send_1d_buf \n");

#else

    if( (send_1d_buf = (float*)calloc_1d_array(buf_size * n_tot_ms_snd, sizeof(float)))==NULL)
    ath_error("[calloc_1d] failed to allocate send_1d_buf \n");

#endif

    if( (segm_mask_send=(int*)calloc_1d_array(n_tot_ms_snd, sizeof(int)))==NULL)
      ath_error("[calloc_1d] failed to allocate segm_mask_send\n");

    /* n_tot_ms_snd =0; */
    
    for(int l=0; l <= NumSegBlock_glb-1; l++){
      pG->RaySegGridPerBlock[l].n_msg_r = 0; //initialize
      /* fill mask arrays and  calc tot number of messages */
      if ( pG->RaySegGridPerBlock[l].type == Head ){	
	n_tot_ms_rcv += (pG->RaySegGridPerBlock)[l].MPI_idConnectArray_size-1;
      }	      
      else if (pG->RaySegGridPerBlock[l].type == Upstr){
	segm_mask_send[l_snd++] = l;
      }      
    }
    /* printf(" case 2 ---- %d, %d ,%d \n", my_id, n_tot_ms_snd, n_tot_ms_rcv); */
    

     for(int ls =0; ls < n_tot_ms_snd ; ls++){

       int l = segm_mask_send[ls];    /* fetch the true ref to segment */   

	 int dest_id = pG->RaySegGridPerBlock[l].head_block[0];
	 
	 if (my_id == dest_id) ath_error("error with type");
        
	 /* for(int i=0; i < num_extra; i++) //pack coords of the head */
	 /*   send_1d_buf[i]= pG->RaySegGridPerBlock[l].head_block[i];	  */
	 /* for(int i=num_extra; i<buf_size; i++){ */
	 /*   send_1d_buf[i]= pG->RaySegGridPerBlock[l].dat_vec_1d[i-num_extra]; */

	 
#ifdef tau_sync_small_buf

	 for(int i=0; i < num_extra; i++) //pack coords of the head
	   send_1d_buf[i]= (float)pG->RaySegGridPerBlock[l].head_block[i];
	 
	 send_1d_buf[0] = (float)n_tot_ms_snd;
	 
	 for(int i=num_extra; i<buf_size; i++)
	   send_1d_buf[i]= pG->RaySegGridPerBlock[l].dat_vec_1d[i-num_extra];

	 int ierr = MPI_Isend(&send_1d_buf[0], buf_size, MPI_FLOAT, dest_id,
	 		     tag_send, MPI_COMM_WORLD, &request);
#else
	   
	 for(int i=0; i < num_extra; i++) //pack coords of the head
	   send_1d_buf[i + ls*buf_size]= (float)pG->RaySegGridPerBlock[l].head_block[i];
	 
	 send_1d_buf[ls*buf_size] = (float)my_id;//(float)n_tot_ms_snd;
	 
	 for(int i=num_extra; i<buf_size; i++)
	   send_1d_buf[i + ls*buf_size]= pG->RaySegGridPerBlock[l].dat_vec_1d[i-num_extra];
	   
	   /* printf("l=%d, data= %f \n", l, send_1d_buf[i]);	    */
	 	 			     
	 int ierr = MPI_Isend(&send_1d_buf[ ls * buf_size ], buf_size, MPI_FLOAT, dest_id,
	 		     tag_send, MPI_COMM_WORLD, &request);

#endif
	 
	 /* int ierr = MPI_Send(&send_1d_buf[ ls * buf_size ], buf_size, MPI_FLOAT, dest_id, */
	 /* 		     tag_send, MPI_COMM_WORLD); */	 
	 /* int ierr = MPI_Send(&send_1d_buf[0], buf_size, MPI_FLOAT, dest_id, */
	 /* 		     tag_send, MPI_COMM_WORLD); */

	 nmsg_snd +=1;


	 /* for(int i=0; i<buf_size; i++) printf("s_id %d ijk=(%d %d %d) data=%f \n", */
	 /*        send_1d_buf[0], send_1d_buf[1], send_1d_buf[2], send_1d_buf[3], send_1d_buf[4] ); */
	
     }


  
     int lr=0;

     
#ifdef mpi_opt_depth_test1      
     float err1  = 0;
#endif
     
   /*   int mpi1=1; */
   /* while(mpi1==1); */
     
     while(lr < n_tot_ms_rcv){

       // printf("receiving %d %d %d\n", l, lr, n_tot_ms_rcv);
	  
       MPI_Recv(&recv_1d_buf[0], buf_size, MPI_FLOAT, MPI_ANY_SOURCE,
		tag_send, MPI_COMM_WORLD, &status);

       if(status.MPI_ERROR==0){
	 nmsg_recv+=1;
	 int id = round(recv_1d_buf[0]);
	 int i = round(recv_1d_buf[1]);
	 int j = round(recv_1d_buf[2]);
	 int k = round(recv_1d_buf[3]);

	 /* printf(" n_msg_sent_ = %d \n",  n_msg_sent_); */
	 
	 int l = pG->BlockToRaySegMask[k][i];
	 pG->RaySegGridPerBlock[l].n_msg_r += 1;

	 
	 for(int m = 0; m < phi_sz; m++)
	   pG->RaySegGridPerBlock[l].dat_vec_1d[m] += recv_1d_buf[m+num_extra];
	 
	 /* for(int m=0; m< phi_sz; m++) */
	 /*    printf("r_id %d ijk=(%d %d %d) %f \n", id, i, j, k, */
	 /* 	   pG->RaySegGridPerBlock[l].dat_vec_1d[m]); */



	   /* int m=k; */
	 /*   printf("r_id %d ijk=(%d %d %d) %f \n", id, i, j, k, */
	 /* 	  pG->RaySegGridPerBlock[l].dat_vec_1d[m] */
	 /* 	  ); */

	 { /* calc the same along the Ray */
#ifdef mpi_opt_depth_test1 	 	 
	   int len = (pG->GridOfRays[k][i]).len;
	   float s=0;
	   for (int n = 0; n < len; n++){
	     s += (pG->GridOfRays[k][i]).Ray[n].dl;
	     //	      	printf("%d, %f \n", ii, s);
	   }
#endif	   

	   int con_arr_sz =  (pG->RaySegGridPerBlock)[l].MPI_idConnectArray_size;	   

	   /* if (n_tot_ms_rcv != con_arr_sz-1){ */
	   /*   printf("n_tot_ms_rcv=%d, con_arr_sz=%d, n_msg_sent_=%d\n", */
	   /* 	    n_tot_ms_rcv, con_arr_sz); */
	   /*   ath_error("error n_tot_ms_rcv,  con_arr_sz-1, n_msg_sent_ problems "); */

	   /* } */

	     
	   if (pG->RaySegGridPerBlock[l].n_msg_r == con_arr_sz-1){	     

	     /* printf("****** id %d n_msg_sent_: %d , actual %d con_arr_ %d \n", my_id, n_msg_sent_, */
	     /* 	    pG->RaySegGridPerBlock[l].n_msg_r,  con_arr_sz); */

	     for(int m = 0; m < phi_sz; m++) {
	       
	       int iloc, jloc, kloc;	       
	       ijkGlobToLoc(pG, i, m, k, &iloc, &jloc, &kloc);

	      if (pG->tau_e == NULL) ath_error("pG->tau_e == NULL, error ");	       	       
	       pG->tau_e[kloc][jloc][iloc] = pG->RaySegGridPerBlock[l].dat_vec_1d[m];
	       
#ifdef mpi_opt_depth_test1 
	       err1 += fabs( pG->RaySegGridPerBlock[l].dat_vec_1d[m] - s );
	       err += fabs(pG->tau_e[kloc][jloc][iloc]-s);
	       // pG->yglob[k][m][i].tau = pG->RaySegGridPerBlock[l].dat_vec_1d[m];
	       pG->yglob[k][m][i].tau = s;     	   
	       /* if(my_id==1) */
	       /* 	 printf("id:%d, src: %d, err=%f %f\n", */
	       /* 		my_id, status.MPI_SOURCE, err1, err); */
#endif
	       
	     }	     	     
	   }	  	 	   
	 }
	 
	 lr+=1;

	 //printf(" n_msg received = %d \n",  lr);
	 
       } else   ath_error("error with MPI_ERROR==0");
       
#ifdef mpi_opt_depth_test1 
     printf("id:%d;  err_s0 =%f \n",  my_id, err);
#endif 
       
     }/* eof receive-loop */


     MPI_Barrier(MPI_COMM_WORLD);

     
      /* special case of the Grid closest to the source  */
     for(int l=0; l <= NumSegBlock_glb-1; l++){      
       if ( pG->RaySegGridPerBlock[l].type == Head && n_tot_ms_rcv == 0){	 
	 for(int m = 0; m < phi_sz; m++) {
	   int i = pG->RaySegGridPerBlock[l].head_block[1];
	   int k = pG->RaySegGridPerBlock[l].head_block[3];

	   float s=0;	   
#ifdef mpi_opt_depth_test1 	   	   
	   for (int n = 0; n < (pG->GridOfRays[k][i]).len; n++)
	     s += (pG->GridOfRays[k][i]).Ray[n].dl;
	   pG->yglob[k][m][i].tau = s;
#endif
	   
           int iloc, jloc, kloc;	       
           ijkGlobToLoc(pG, i, m, k, &iloc, &jloc, &kloc);

	   pG->tau_e[kloc][jloc][iloc] = pG->RaySegGridPerBlock[l].dat_vec_1d[m]; 

	   //	   pG->yglob[k][m][i].tau = pG->tau_e[kloc][jloc][iloc];
	   //	   pG->yglob[k][m][i].tau = pG->tau_e[kloc][jloc][iloc];	   
	   /* printf("%f \n", pG->yglob[k][m][i].tau); */	   
	   /* err += fabs(  pG->tau_e[kloc][jloc][iloc] -  pG->yglob[k][m][i].tau); */
	   // printf("id0:%d;  res1 =%f res2=%f\n",  my_id, pG->tau_e[kloc][jloc][iloc], s);
	 }	 
       }
     }

     //MPI_Barrier(MPI_COMM_WORLD);
     /* printf("%d total sent %d; received %d \n", my_id,  nmsg_snd, nmsg_recv); */
     
#ifdef mpi_opt_depth_test1 	 	      
     //printf("id:%d;  err_s1 =%f \n",  my_id, err);                               	
     int kp, jp , ip;
     for (kp = BufBndr[my_id][2]; kp <= BufBndr[my_id][5]; kp++) {
       for (jp = BufBndr[my_id][1]; jp<=BufBndr[my_id][4]; jp++) {
	 for (ip =BufBndr[my_id][0]; ip<=BufBndr[my_id][3]; ip++) {
	   int iloc, jloc, kloc;
	   ijkGlobToLoc(pG, ip, jp, kp, &iloc, &jloc, &kloc);
	   /* 	     err += fabs(pG->yglob[kp][jp][kp].tau - pG->tau_e[kloc][jloc][iloc]); */
	   pG->yglob[kp][jp][ip].tau = pG->tau_e[kloc][jloc][iloc];
	   /* printf("id %d  %d %d %d\n",my_id, ip, kp, jp);  */
	   /* 	     /\* if(my_id==0) *\/ */
	   /* 	     /\* printf("id %d tau_g=%f, tau_l=%f ijk= (%d %d %d)\n", *\/ */
	   /* 	     /\* 	    my_id, *\/ */
	   /* 	     /\* 	    pG->yglob[kp][jp][kp].taru, pG->tau_e[kloc][jloc][iloc], *\/ */
	   /* 	     /\* 	    ip, kp, jp); *\/ */
	 }
       }
     }
     SyncGridGlob(pM, pG, ID_TAU);	
#endif
     
/* #ifdef mpi_opt_depth_test1  */
/*        // printf("id %d err_end = %f \n", my_id, err); */
/* #endif */


   MPI_Barrier(MPI_COMM_WORLD);

    
   //if(my_id == 0)

      //aplot(pM, 0,0,0, pM->Nx[0]-1, pM->Nx[1]-1, pM->Nx[2]-1, "tau");
    
      /* aplot(pM, 0,0,0, pM->Nx[0]-1, pM->Nx[1]-1, pM->Nx[2]-1, "ro"); */
    break;
	
  default:
    printf("%d \n", TestType);
    ath_error("unknown case in testSegments2");
    
  } // switch
 
  
  free_1d_array(segm_mask_send);
  free_1d_array(recv_1d_buf);
  free_1d_array(send_1d_buf);
  /* free_2d_array(send_2d_buf); */
}


/* void CalcParallelOdepthData(GridS *pG, int my_id){ */

/*   int ig_s = BufBndr[my_id][0]; */
/*   int ig_e = BufBndr[my_id][3]; */
/*   int kg_s = BufBndr[my_id][2]; */
/*   int kg_e = BufBndr[my_id][5]; */
        	
/*     {//create MPI datatype */
/*       const int nitems=3; */
/*       int blocklengths[3] = {1,1,1};	   */
/*       MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_FLOAT}; */
/*       MPI_Datatype mpi_seg_data;	  */
/*       MPI_Aint offsets[3]; */

/*       offsets[0] = offsetof(SegHeadId, i); */
/*       offsets[1] = offsetof(SegHeadId, k); */
/*       offsets[2] = offsetof(SegHeadId, dat);	   */

/*       MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_seg_data); */
/*       MPI_Type_commit(&mpi_seg_data); */
	
	
/*       SegHeadId SegSendBuf, SegRecvBuf; */
	  
/*       const int tag = 13; */


/*       for (int l=0; l <= NumSegBlock_glb - 1; l++){	 */

/* 	int dest_id = pG->RaySegGridPerBlock[l].mpi_id_ik_head_block[0];	 */

/* 	int mpi1=1; */
/* 	/\* while(mpi1==1); *\/				  		     */

/* 	int i_loc, k_loc, l_loc; */
  
/* 	if (my_id != dest_id){ */

/* 	  SegSendBuf.i = pG->RaySegGridPerBlock[l].mpi_id_ik_head_block[1]; */
/* 	  SegSendBuf.k = pG->RaySegGridPerBlock[l].mpi_id_ik_head_block[2]; */
/* 	  SegSendBuf.dat = pG->RaySegGridPerBlock[l].dTau; */

/* 	  if(pG->RaySegGridPerBlock[l].type == Head) ath_error("error in testSegments2"); */
	    
/* 	  MPI_Send(&SegSendBuf, 1, mpi_seg_data, dest_id, tag, MPI_COMM_WORLD); */

/* 	  // printf("send: %d, %d, %f \n", SegSendBuf.i, SegSendBuf.k, SegSendBuf.dat); */
/* 	}   */
	  

/* 	if (my_id == dest_id){ */
	    
/* 	  MPI_Status status; */

/* 	  /\* we dont need id of my_id, i.e. relevant size = real_size-2 *\/	     */
/* 	  int buf_max =  (pG->RaySegGridPerBlock)[l].MPI_idConnectArray_size; */
	    
/* 	  for(int impi=0; impi <= buf_max-2; impi++){ */
/* 	    /\* iterate over upstream segments, current elem not included *\/ */

/* 	    int orig_id = pG->RaySegGridPerBlock[l].MPI_idConnectArray[impi]; */
/* 	    //printf("MPI connect, my_id=%d, mpi_indx=%d \n", my_id,mpi_indx); */

/* 	    MPI_Recv(&SegRecvBuf, 1, mpi_seg_data, orig_id, tag, MPI_COMM_WORLD, &status); */
	      
/* 	    /\* printf("from:%d to:%d, %d, %d %f \n", orig_id, dest_id, SegRecvBuf.i, *\/ */
/* 	    /\* 	     SegRecvBuf.k, SegRecvBuf.dat); *\/ */

/* 	    k_loc=SegRecvBuf.k; */
/* 	    i_loc=SegRecvBuf.i;	       */
/* 	    l_loc=pG->BlockToRaySegMask[k_loc][i_loc]; */

/* 	    /\* printf("id = %d, lloc= %d \n", my_id, l_loc); *\/ */
		
/* 	    pG->RaySegGridPerBlock[l_loc].dTau += SegRecvBuf.dat; */
	     	      
/* 	  } */
	    
/* 	  if (buf_max < 2){//closest to source	       */
/* 	    i_loc=pG->RaySegGridPerBlock[l].mpi_id_ik_head_block[1]; */
/* 	    k_loc=pG->RaySegGridPerBlock[l].mpi_id_ik_head_block[2]; */
/* 	    l_loc=pG->BlockToRaySegMask[k_loc][i_loc]; */
	       
/* 	  } */
	    
/* 	  /\* calc the same along the Ray *\/ */
/* 	  Real s=0; */
/* 	  int len=(pG->GridOfRays[k_loc][i_loc]).len; */

/* 	  for (int ii = 0; ii<len; ii++){ */
/* 	    s += (pG->GridOfRays[k_loc][i_loc]).Ray[ii].dl; */
/* 	    //	      	printf("%d, %f \n", ii, s); */
/* 	  } */

/* 	  printf("id:%d, s=%f, dat=%f \n",my_id, s, pG->RaySegGridPerBlock[l_loc].dTau); */
	    	    
/* 	} /\* my_id == dest_id *\/ */
	  	  
/*       }	  	 */
/*     }/\* end seg-loop *\/    	      */


/* } */




  

void Constr_SegmentsFromRays_version1(MeshS *pM, GridS *pG, int my_id){
  /*! breaks the grid of rays, obtained from global grid into individual segments
    belonging to a particular MPI Block = Grid */
  
  int ig,ig_s,ig_e, kg,kg_s,kg_e, jg, je,ii, i,k,NumSeg;
  int NumOfSegInBlock,NumMPIBlocks,cur_id,pr_id,mpi1;
  enum BCBufId Side;
  /* enum WhereIAM {InBlock=0, OutBlock=1}InOutBlock; */     
  int *tmp_mpiIdVec;        //tmp stores mpi ids of segments
  enum SegmentType *tmp_SegType;   //tmp stores array of segments type 
  RaySegment   *tmpSegGrid; /* temp array: we don't know how many segs per Block */ 
 
  ig_s = BufBndr[my_id][0];
  ig_e = BufBndr[my_id][3];
  kg_s = BufBndr[my_id][2];
  kg_e = BufBndr[my_id][5];

  MPI_Comm_size(MPI_COMM_WORLD, &NumMPIBlocks);
  
  int N = (ig_e-ig_s)*(kg_e-kg_s);
  int N2 = N*N;
  if( (tmp_mpiIdVec = (int*)calloc_1d_array(N2, sizeof(int)))==NULL)    
      ath_error("[calloc_1d] failed to allocate mpiIdVec \n");
  
  if( (tmpSegGrid = (RaySegment*)calloc_1d_array(N2, sizeof(RaySegment)))==NULL)
      ath_error("[calloc_1d] failed to allocate tmpRaySegGridPerBlock \n");
     
  jg = 0; /* rays must be in phi_j =const planes */
    
  for (kg=kg_s; kg<=kg_e; kg++) {   // z
    for (ig=ig_s; ig<=ig_e; ig++) { //R
      
      
      findSegmPartitionFromRay(pG, pG->GridOfRays[kg][ig], ig,kg,
			       tmpSegGrid, &NumSeg, tmp_mpiIdVec,
			       NumMPIBlocks, my_id);
      /* mpi1=1; */
      /* while( mpi1==1 ); */

      testSegments1(pG, ig, kg, tmpSegGrid, my_id);
	
      /* printf("segment test1, id: %d , ig= %d , kg= %d  \n", my_id, ig, kg); */
      
    } //ig
  } //kg

  free(tmp_mpiIdVec);

  mpi1=1;
  while( mpi1==1 );
}



void testSegments1(GridS *pG, int ip, int kp, RaySegment* tmpSegGrid, int my_id){
  /* first prints the full ray; then the segments */

  int ii,itr;
  float s1=0,s2=0;
  int s11=0,s22=0;
  
  int len=(pG->GridOfRays[kp][ip]).len;
  
  /* mpi1=1; */
  /* while( mpi1==1 ); */      

  {    
  for (ii = 0; ii<len; ii++){
    s1 += (pG->GridOfRays[kp][ip]).Ray[ii].dl;
    s11 += (pG->GridOfRays[kp][ip].Ray[ii].i + pG->GridOfRays[kp][ip].Ray[ii].k);
  }

  
  int ns=0;      
  for (int l=0; l<= 1000; l++){
    
    for(int i=0; i<=tmpSegGrid[l].data_len-1;i++){
      s2 += tmpSegGrid[l].data[i].dl;
      s22 += (tmpSegGrid[l].data[i].i +tmpSegGrid[l].data[i].k);	
    }    
    ns++;
    if (tmpSegGrid[l].type == Head) break;
  }
  printf("id=%d ip=%d, kp=%d; ns=%d, s1=%f s2=%f, s11=%d s22=%d \n",
	 my_id, ip, kp, ns, s1, s2, s11, s22);
  printf("==============================================\n");
  }

}  


void findSegmPartitionFromRay(GridS *pG, RayData GridOfRays,int ig,int kg,
			      RaySegment* tmpSegGrid,
			      int* NumSeg,
			      int* tmp_mpiIdVec,
			      int NumMPIBlocks, int my_id){
  int ii,i,j,k,ip,seg_i,cur_id,prev_id,fut_id,is_s,is_e;

  cur_id = -1;
  prev_id = -1;
  fut_id = -1;
  seg_i = 0;

  /* mpi1=1; */
  /* while(mpi1==1 ); */
  
  for (ii = 0; ii< GridOfRays.len; ii++){
  
    SearchBufId(GridOfRays.Ray[ii].i, GridOfRays.Ray[ii].k, NumMPIBlocks, &cur_id);
    
    tmpSegGrid[seg_i].type = Upstr; //default val
    tmpSegGrid[seg_i].MPI_idConnectArray_size =0; //default val

    if(ii==0){ //very first seg; first elem */
      tmpSegGrid[seg_i].data_len=1;
      tmp_mpiIdVec[seg_i] = cur_id;
      is_s = ii;
      is_e = is_s;      
      prev_id = cur_id;      
    }

    /* look ahead */
    if (ii< GridOfRays.len-1){
      SearchBufId(GridOfRays.Ray[ii+1].i, GridOfRays.Ray[ii+1].k, NumMPIBlocks, &fut_id);
    }

    if((fut_id != cur_id) || (ii==GridOfRays.len-1)){/* last cell of a segment */           
      is_e = ii;
      tmpSegGrid[seg_i].data_len = ii-is_s+1;      
      tmp_mpiIdVec[seg_i] = cur_id;
      
      if( (tmpSegGrid[seg_i].data=
      	   (SegmentData*)calloc_1d_array(tmpSegGrid[seg_i].data_len,sizeof(SegmentData)))==NULL)
	   ath_error("[calloc_1d] failed to allocate findSegmPartitionFromRay, 1 \n");

      if (ii==GridOfRays.len-1){

	tmpSegGrid[seg_i].type = Head;
	tmpSegGrid[seg_i].MPI_idConnectArray_size = seg_i+1;
	
	if( (tmpSegGrid[seg_i].MPI_idConnectArray=
	     (int*)calloc_1d_array(seg_i+1,sizeof(int)))==NULL)
	  ath_error("[calloc_1d] failed to allocate findSegmPartitionFromRay, 2 \n");
	for(i=0; i<=seg_i; i++) tmpSegGrid[seg_i].MPI_idConnectArray[i]=tmp_mpiIdVec[i];
      }
      
      for(ip = is_s; ip<= is_e; ip++){ //is_s and is_e are in the Ray structure
	int l = ip - is_s;
	tmpSegGrid[seg_i].data[l].i = GridOfRays.Ray[ip].i;	
	tmpSegGrid[seg_i].data[l].k = GridOfRays.Ray[ip].k;
      	tmpSegGrid[seg_i].data[l].dl = GridOfRays.Ray[ip].dl;	
      }
     
      is_s = ii+1; //next first cell          
      seg_i+=1;
    } /* end last cell of a segment */


    }//ii

  *NumSeg=seg_i;
}


/* void testSegments1(GridS *pG, int ip, int kp, RaySegment* tmpSegGrid, int my_id){ */
/*   /\* first prints the full ray; then the segments *\/ */

/*   int ii,itr; */
/*   float s1=0,s2=0; */
/*   int s11=0,s22=0; */
  
/*   int len=(pG->GridOfRays[kp][ip]).len; */
  
/*   /\* mpi1=1; *\/ */
/*   /\* while( mpi1==1 ); *\/       */

/*   {     */
/*   for (ii = 0; ii<len; ii++){ */
/*     s1 += (pG->GridOfRays[kp][ip]).Ray[ii].dl; */
/*     s11 += (pG->GridOfRays[kp][ip].Ray[ii].i + pG->GridOfRays[kp][ip].Ray[ii].k); */
/*   } */

  
/*   int ns=0;       */
/*   for (int l=0; l<= 1000; l++){ */
    
/*     for(int i=0; i<=tmpSegGrid[l].data_len-1;i++){ */
/*       s2 += tmpSegGrid[l].data[i].dl; */
/*       s22 += (tmpSegGrid[l].data[i].i +tmpSegGrid[l].data[i].k);	 */
/*     }     */
/*     ns++; */
/*     if (tmpSegGrid[l].type == Head) break; */
/*   } */
/*   printf("id=%d ip=%d, kp=%d; ns=%d, s1=%f s2=%f, s11=%d s22=%d \n", */
/* 	 my_id, ip, kp, ns, s1, s2, s11, s22); */
/*   printf("==============================================\n"); */
/*   } */

/* } */

  
void SearchBufId(int i, int k, int NumMpiBlocks, int* id_res){
  /* finds MPI block (Grid) where we are located; (R,Z) plane only*/
  int  is,ie,ks,ke,id;
  
  for(id=0; id < NumMpiBlocks; id++){
      is = BufBndr[id][0];
      ie = BufBndr[id][3];
      ks = BufBndr[id][2];
      ke = BufBndr[id][5];
      
      if((i>=is) && (i<=ie) && (k>=ks) && (k<=ke)){
	*id_res = id;
	break;
      }
  }        
}





enum BCBufId CheckCrossBlockBdry(int i, int k, int *BlockId){
   /* checks if the ray passes block side */
  int is,ie,ks,ke;
 
  is = BufBndr[*BlockId][0];
  ie = BufBndr[*BlockId][3];
  ks = BufBndr[*BlockId][2];
  ke = BufBndr[*BlockId][5];  

  if (i==is) return LS;
  if (i==ie) return RS;
  if (k==ks) return DS;
  if (k==ke) return US;

  return NotOnSide; /* default val */   
}


CellOnRayData *tmpCellIndexAndDisArray;

int ncros;


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

	if ((pG->U[kp][jp][ip].d > tiny)){
	  cc_pos(pG,ip,jp,kp, &x1,&x2,&x3);
	  
	  xi = Lx / (Nsc* fmax(pG->U[kp][jp][ip].d , rho0))/
	    pow(Rsc,2)/fmax(pow(x1 - rRadSrc[0], 2) + pow(x3 - rRadSrc[1] ,2), tiny);
	  /* printf("%f\n", Nsc); */
	}	
	else{
	  printf("quite possibly something is negative in opticalProps ");
	  /* apause(); */
	}

	pG->xi[kp][jp][ip]  = xi;
	
#ifdef save_memory

	Real tau_x= pG->tau_e[kp][jp][ip];

	/* printf(" %f \n", tau_x); */
	
#else
	/* position on glob */
	ijkLocToGlob(pG, ip,jp,kp, &i_g, &j_g, &k_g);
	Real tau_x = pG->yglob[k_g][j_g][i_g].tau;
#endif		
	pG->xi[kp][jp][ip]  *=  exp(-tau_x);
	/* printf("%f %f \n", tau_x, pG->xi[kp][jp][ip] ); */

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


#ifdef CAK_FORCE
Real CAK_RadPresLines(ConsS ***U, GridS *pG, const Real xi,
		      const Real dx1,  const Real dx2,  const Real dx3,		      
		      const Real dt, int i, int j, int k)
{
  Real eta_max, Mt_max, Press, Pnew, Tg, k1=0.03,vth,t, dvdx[3];
  const Real A=1, alp = ;
  
  Real d = U[k][j][i].d;
  Real di = 1./d, Ujik;

  Ujik = U[k][j][i].M1/U[k][j][i].d
  
  dvdx[0] = (U[k][j][i+1].M1/U[k][j][i+1].d - Ujik)/dx1;
  
  dvdx[1] = (U[k][j+1][i].M1/U[k][j+1][i].d - Ujik)/dx2;

  dvdx[2] = (U[k+1][j][i].M1/U[k+1][j][i].d - Ujik)/dx3;

  
  
  Press = U[k][j][i].E -  0.5*(SQR(U[k][j][i].M1) + SQR(U[k][j][i].M2) + SQR(U[k][j][i].M3))*di;
  
  Press -=  0.5*(SQR(U[k][j][i].B1c) + SQR(U[k][j][i].B2c) + SQR(U[k][j][i].B3c));
  
  Press *=  Gamma_1;
  
  Tg =  Pnew * Esc * M_MW / (U[k][j][i].d*Dsc*RGAS );

  vth=(12.85e+5)*sqrt(Tg /1.e4/A);

  /* vxl = ML*di; */
  /* vxr = MR*di;   */
    
  k1 = 0.03 + 0.385*exp(-1.4* pow(xi,0.6));

  if ( xi < 3.16){
    eta_max= pow(10., 6.9 * exp( 0.16*pow(xi, 0.4) ) );
  }
  else { 
    eta_max= pow(10., 9.1 * exp(-7.96e-3 * xi) );
  }
  
  // taumax=t * eta_max

  Mt_max=(1. - alp ) * k1 * pow(eta_max, alp);

  dvdx1 = dvdx1 *Usc/Rsc;

  ro=d *MP;
  
  t=k1 * KPE * vth * ro  / max(dvdx1, tiny);


    kk = ( (1.+taumax)**(1.- alp)-1.)/taumax**(1.-alp)

    Mt=k1*kk * t**(-alp)

    if (Mt > Mt_max) Mt=Mt_max
		       //	  k1=0.03

    //	  rij = (/ grid1% x1a(i), grid1% x2a(j) /)

	  rsph =sqrt ( dot_product( rij, rij)  )

	  grdv = (/ (v1(i,j,ks) - v1(i-1,j, ks) ) * dx1bi(i),
		  (v2(i,j,ks) - v2(i,j-1,ks) ) * dx2bi(j) /)

	  dvdx1 = dvdx1 *Usc/Rsc

	  Tg = gas1%Tg(i,j)
	 
	  ndens=max( gas1% dns(i, j), tiny)*nc0

	  ro=ndens*mp
      
}
#endif /* CAK_FORCE */


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

      /* printf(" Tg = %e dens = %e \n", Tg, d*Dsc); */
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
	fprintf(f, "%f \n", log10(pG->xi[k][j][i]) );
      }
#endif
    }
    fprintf(f,"\n");
  }
  fclose(f);
  system("./plot_from_athena.py");
}

#ifdef XRAYS
#ifdef use_glob_vars 
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
  printf("phi slice, j_phi = %d \n", j );
  //printf(" %d %d \n", is, ie);
  
  fprintf(f, "%d  %d\n", nx1, nx3 );
  
  for (k=kl; k<=ku; k++) {
    for (i=il; i<=iu; i++){
      if (strcmp(name, "tau") == 0){
	//fprintf(f, "%f\n",  log10(pG->yglob[k][j][i].tau) );
		fprintf(f, "%f\n",  (pG->yglob[k][j][i].tau) );
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
#endif 
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


/* problem
:   */
//#define VP(R) (  (sqrt(r0)/(r0-2)) * pow(R/r0,-q+1)  )

//#define VP(R) pow(xm,q)*pow(R, 1.-q)


#define VP(R) pow(R/xm, 1.-q)


void problem(MeshS *pM, DomainS *pDomain)
{
  GridS *pG=(pDomain->Grid);


  
#ifdef XRAYS
  CoolingFunc = updateEnergyFromXrayHeatCool; 
#endif
#ifdef CAK_FORCE
 RadiationPresLines = CAK_RadPresLines;
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
  
  /* 1) */


  //{ printf("MPI breakpoint, id=%d \n", my_id); int mpi1= 1;  while(mpi1 == 1);}   
  Constr_RayCastingOnGlobGrid(pM, pG, my_id); //mainly calc. GridOfRays
 

  /* 2) */
  initGlobComBuffer(pM, pG);  //here is the problem
  
  MPI_Barrier(MPI_COMM_WORLD);


  
  /* sync global patch */  
#ifdef use_glob_vars 
  SyncGridGlob(pM, pG, ID_DEN);
   MPI_Barrier(MPI_COMM_WORLD);
#endif
  
 
  
  { /*  ------- optical depth block from segments in parallel------------*/
    /* 1) construct grid of ray segments which belong to MPI-block; */
    /* unique: by index of the ray on a block; id,i,k) id and coords of the ray ending */

    my_id_glob=my_id;
     
    Constr_SegmentsFromRays(pM, pG, my_id);    
    MPI_Barrier(MPI_COMM_WORLD);

  /* { printf("MPI breakpoint, id=%d \n", my_id); int mpi1= 1;  while(mpi1 == 1);}   */
    
    OptDepthAllSegBlock(pG); // tau on a local mpi block
    testSegments2(pM, pG, my_id, 2);    //MPI sync tau segments
    MPI_Barrier(MPI_COMM_WORLD);
    
  
   
}

#ifdef use_glob_vars    
  //optDepthStackOnGlobGrid(pM, pG, my_id); //calc. tau
  //SyncGridGlob(pM,pG, ID_TAU);           
#endif
      
   ionizParam(pM, pG);
   MPI_Barrier(MPI_COMM_WORLD);
  /* if(my_id == 0) plot(pM, "xi");   */



   
  /* calcGravitySolverParams(pM, pG); */

 
#else  /* NOT PARALLEL *cd/

    /* printf("  pM->RootMinX[0] = %f \n\n", pM->RootMinX[0] ); */    
    // Constr_optDepthStackOnGlobGrid(pM, pG); //mainly calc. GridOfRays    
    // optDepthStackOnGlobGrid(pM, pG); //calc. tau

    /* ionizParam(pM, pG); */
    
    // Constr_optDepthStack(pM, pG);
    // optDepthStack(pM, pG);
    
#endif /* MPI */


//getchar();   
// 
   /* bvals_tau(pDomain); */
   /* getchar(); */

#endif /* XRAYS */





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

    /* while(mpi1 == 1); */
    
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

void problem_read_restart(MeshS *pM,  GridS *pG)
/* void problem_read_restart(GridS *pG, DomainS *pD, FILE *fp) */
{

  DomainS *pDomain = (DomainS*)&(pM->Domain[0][0]);
  /* GridS *pG = pD->Grid; */

  if (pG->GridOfRays == NULL) ath_error("pG->GridOfRays != NULL, error on restart");

  
  /* (pG->GridOfRays[0][0]).Ray = */

    
  /* (pG->GridOfRays[0][0]).Ray = */
  /* 	(CellOnRayData*)calloc_1d_array(1,sizeof(CellOnRayData)); */
  
  /* if (pM->Domain[0][0].Grid != NULL) { */
  /*   pG = pM->Domain[0][0].Grid;   				     */
  /* } else { */
  /*   ath_error("error 1 on restart");     */
  /* } */          
  int my_id;
  
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
  
  /* Enroll the gravitational function and radial BC */

  if(MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD, &my_id))
    ath_error("[main]: Error on calling MPI_Comm_rank in torus9.c\n");

#ifdef XRAYS
  CoolingFunc = updateEnergyFromXrayHeatCool; 
#endif
#ifdef CAK_FORCE
  RadiationPresLines = CAK_RadPresLines;
#endif

  StaticGravPot = grav_pot;
  x1GravAcc = grav_acc;

  calcProblemParameters();  
  printProblemParameters();
  
  bvals_mhd_fun(pDomain, left_x1,  inX1BoundCond );  
  bvals_mhd_fun(pDomain, left_x3,  diode_outflow_ix3 );
  bvals_mhd_fun(pDomain, right_x3, diode_outflow_ox3);

  
  MPI_Barrier(MPI_COMM_WORLD);

  /* pG->GridOfRays = (RayData**)calloc_2d_array(pM->Nx[2], pM->Nx[0], sizeof(RayData)); */
  //if (pG->GridOfRays == NULL) goto on_error_xrays_GridOfRays;      
  
  /* 1) */
  Constr_RayCastingOnGlobGrid(pM, pG, my_id); //mainly calc. GridOfRays

 
  
  
  /* 2) */
  initGlobComBuffer(pM, pG);  //here is the problem
  /* 3) */
  MPI_Barrier(MPI_COMM_WORLD);
  /* 4) */

#ifdef use_glob_vars
  SyncGridGlob(pM, pG, ID_DEN);
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  my_id_glob=my_id;

  Constr_SegmentsFromRays(pM, pG, my_id);    


  MPI_Barrier(MPI_COMM_WORLD);

  /* { printf("MPI breakpoint, id=%d \n", my_id); int mpi1= 1;  while(mpi1 == 1);}   */

    
  OptDepthAllSegBlock(pG); // tau on a local mpi block


  testSegments2(pM, pG, my_id, 2);    //MPI sync tau segments
  MPI_Barrier(MPI_COMM_WORLD);


  ionizParam(pM, pG);
  MPI_Barrier(MPI_COMM_WORLD);

    /* { printf("MPI breakpoint2, id=%d \n", -20); int mpi1= 1;  while(mpi1 == 1);}  */
  
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

  //  { printf("MPI breakpoint2, id=%d \n", -20); int mpi1= 1;  while(mpi1 == 1);}
    
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

#ifdef use_glob_vars 
  SyncGridGlob(pM, pG, ID_DEN);
#endif

  MPI_Barrier(MPI_COMM_WORLD);
  
  /* optDepthStackOnGlobGrid(pM, pG, my_id);   */
  /* SyncGridGlob(pM, pG, ID_TAU);   */
 
   
  OptDepthAllSegBlock(pG); // tau on a local mpi block
  testSegments2(pM, pG, my_id, 2);    //MPI sync tau segments
  MPI_Barrier(MPI_COMM_WORLD);  
  ionizParam(pM, pG);

  MPI_Barrier(MPI_COMM_WORLD);


  /* { printf("MPI breakpoint, id=%d \n", my_id); int mpi1= 1;  while(mpi1 == 1);} */
 
  /* plot(pM, "tau"); */
  /* printf("+++++++++++++++++++++\n"); */
 

#ifdef sgHYPRE /**********  SELF-GRAVITY ***********/
  { /* testHYPRE_1(pM, pG); */
    testHYPRE_ex2(pM, pG);

    testSelfGravFourier(pM, pG);

    printf("testHYPRE_ex2(pM, pG); done.. \n");

    int mpi1=1; 
    while(mpi1 == 1);
  }
#endif        /**********  SELF-GRAVITY ***********/
    
#else /* not parallel */




 



 // optDepthStack(pM, pG); 
 //  optDepthStackOnGlobGrid(pM, pG);
 //  ionizParam(pM, pG);
  
   


   
   /* plot(pM, "xi");    */

#endif /* MPI */


#endif /* XRAYS */
      


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
  int i,j,k,is,ie,js,je,ks,ke,kp,jp,ip;
  GridS *pG = pM->Domain[0][0].Grid;
  //DomainS pD = pM->Domain[0][0];		
  /* is = pG->is; */
  /* ie = pG->ie; */
  /* js = pG->js; */
  /* je = pG->je; */
  /* ks = pG->ks; */
  /* ke = pG->ke; */

  is = 0;
  ie = pM->Nx[0]-1;
  js = 0;
  je = pM->Nx[1]-1;       
  ks = 0;
  ke = pM->Nx[2]-1;
	  
#ifdef XRAYS
#ifndef save_memory
    for (kp = ks; kp<=ke; kp++) { //z
      //  for (jp = js; jp<=je; jp++) { //t
      for (ip = is; ip<=ie; ip++) { //r
	free_1d_array((pG->GridOfRays[kp][ip]).Ray);
      }
    }

  free_2d_array(pG->GridOfRays);
#endif	  
#endif

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
    int nroot;

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
			     int* nroot ) {
  /* ijk_cur is the 3d index of the current cell */
  /*  xpos is the (x,y,z) exact cart. position, should be located at one of the cell's boundaries; */
  /*  x1x2x3 is the 3d - r,t,z -center coordinates of the cell to which the 
      above boundaries belong */
  // traces a cingle cell; cnorm is in Cart. coordinates;

  int crosPhiConstPlanes =0;
  int change_indx, icur, jcur, kcur;

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
		   int* nroot) {
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
  *px2 = pM->RootMinX[1] + ((Real)(jloc -pG->js + pG->Disp[1]) + 0.5)*pG->dx2;
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
  /* MPI_Request request; */
  
  int  my_id, im, jm, km, size;
  int is,ie,js,je,ks,ke;
 
  MPI_Comm_size(MPI_COMM_WORLD, &NumDomsInGlob);
 
   is = pG->is; ie = pG->ie;
   js = pG->js; je = pG->je;
   ks = pG->ks; ke = pG->ke;
    
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

 
   int send_buf[6];
   int recv_buf[6], **recv_buf_2d=NULL, **send_buf_2d=NULL;

   if((BufBndr = (int**)calloc_2d_array(NumDomsInGlob, 6, sizeof(int) )) == NULL)
      ath_error("[initGlobBuffer]: Failed to allocate BufBndr buffer\n");

   if((recv_buf_2d = (int**)calloc_2d_array(NumDomsInGlob, 6, sizeof(int) )) == NULL)
      ath_error("[initGlobBuffer]: Failed to allocate recv_buf_2d buffer\n");

   if((send_buf_2d = (int**)calloc_2d_array(NumDomsInGlob, 6, sizeof(int) )) == NULL)
      ath_error("[initGlobBuffer]: Failed to allocate send_buf_2d buffer\n");

   MPI_Reduce(&ie, &im, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);   
   MPI_Reduce(&je, &jm, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&ke, &km, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

   
 
   /* calculate index position of the local patch in global mesh   */
   /* is -> BufBndr1[0]..ks -> BufBndr1[2]; ie->BufBndr1[2].. */

   /* indices in virtual global buffer */
   ijkLocToGlob(pG, is, js, ks,  &(send_buf[0]), &(send_buf[1]), &(send_buf[2]));
   ijkLocToGlob(pG, ie, je, ke,  &(send_buf[3]), &(send_buf[4]), &(send_buf[5]));

int mpi1=1;
//while( mpi1==1 );
   
   for(int m = 0; m<= 5; m++) BufBndr[my_id][m] = send_buf[m];

   if (my_id == 0) {
     BufSize =  im*jm*km;
     ibufe = im;
     jbufe = jm;
     kbufe =km;
     
     for (int id_from =1; id_from < NumDomsInGlob; id_from++){      

       MPI_Recv(&recv_buf, 6, MPI_INT, id_from, 1, MPI_COMM_WORLD, &status);
       
       
       for(int m = 0; m<= 5; m++) BufBndr[id_from][m] = recv_buf[m];       

       /* printf("recv from %d %d %d \n", id_from, status, NumDomsInGlob);      	  */
     }

 
     int *pSnd;
     //pSnd = (int*)&(send_buf_2d[0]);

     pSnd = (int*)&(send_buf_2d[0][0]);

     
 
     for (int i=0; i<NumDomsInGlob; i++){ //pack the send buffer
       for(int ii = 0; ii<=5; ii++){	
	 *(pSnd++) = BufBndr[i][ii];	 
       }
     }

     
      
     for(int id_to=1; id_to<NumDomsInGlob; id_to++) {//send to all else
       /* MPI_Send(&(send_buf_2d[0]),  NumDomsInGlob*6, MPI_INT, id_to, 2, */
       /* 	      MPI_COMM_WORLD); */
       MPI_Send(&(send_buf_2d[0][0]),  NumDomsInGlob*6, MPI_INT, id_to, 2,
     	      MPI_COMM_WORLD);       
     }

  
     
   }
   else{
     int id_to=0;      
     MPI_Send(&send_buf[0], 6, MPI_INT, id_to, 1 , MPI_COMM_WORLD);

     int id_from = 0;
     /* int *pRcv = (int*)&(recv_buf_2d[0]); */
     int *pRcv = (int*)&(recv_buf_2d[0][0]);
     
     /* MPI_Recv(&(recv_buf_2d)[0],  NumDomsInGlob*6, MPI_INT, id_from, 2, */
     /* 	      MPI_COMM_WORLD, &status); */
     MPI_Recv(&(recv_buf_2d)[0][0],  NumDomsInGlob*6, MPI_INT, id_from, 2,
     	      MPI_COMM_WORLD, &status);

     for (int i=0; i<NumDomsInGlob; i++){
    	 for(int ii = 0; ii<=5; ii++){
     	   BufBndr[i][ii] = *(pRcv++);
	 }
     }
     /* printf("recieved from %d %d\n", id_from, status); */
   }
     
   
   MPI_Barrier(MPI_COMM_WORLD);
   for(int ii=0; ii < NumDomsInGlob; ii++){
   int ii=my_id;
   printf(" ijk_s:  %d %d %d, ijk_e: %d %d %d my_id= %d \n\n",
	  BufBndr[ii][0], BufBndr[ii][1], BufBndr[ii][2],
	  BufBndr[ii][3], BufBndr[ii][4], BufBndr[ii][5],
	  ii);
   }
    
   free(recv_buf_2d);
   free(send_buf_2d);

   //   broadcasting to all about global buffer
   MPI_Bcast(&BufSize, 1, MPI_INT,   0, MPI_COMM_WORLD);
   MPI_Bcast(&ibufe,   1, MPI_INT,   0, MPI_COMM_WORLD);
   MPI_Bcast(&jbufe,   1, MPI_INT,   0, MPI_COMM_WORLD);
   MPI_Bcast(&kbufe,   1, MPI_INT,   0, MPI_COMM_WORLD);

#ifdef use_glob_vars
   AllocateSendRecvGlobBuffers(pM, pG);        
#endif
   
   return;
}

#ifdef use_glob_vars

void AllocateSendRecvGlobBuffers(const MeshS *pM, const GridS *pG )
{
/* big 1d bufer for sending yglob */

  BufSizeGlobArr = pM->Nx[2]* pM->Nx[1]* pM->Nx[0];

  printf( "BufSizeGlobArr= %d\n", BufSizeGlobArr);

  /* int  mpi1=1; */
  /* while( mpi1==1 ); */
 
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

}

 
void SyncGridGlob(MeshS *pM, GridS *pG, int W2Do)
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
#endif

#ifdef use_glob_vars
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
#endif

void freeGlobArrays()
{ 
  free_1d_array(recv_rq);
  free_1d_array(send_rq);
  free_2d_array(BufBndr);

#ifdef use_glob_vars
  free_1d_array(send_buf);
  free_1d_array(send_buf_big);
  free_1d_array(recv_buf);
  free_1d_array(recv_buf_big);
#endif
  
}

#endif


/* given x, returns containing cell first index.  */
int celli_Glob(const MeshS *pM, const Real x, const Real dx1_1, int *i, Real *a, const int is)
{
  *a = (x -  pM->RootMinX[0]) * dx1_1 + (Real)is;
  *i = (int)(*a);
  if (((*a)-(*i)) < 0.5) return 0;	/* in the left half of the cell*/
  else return 1;			/* in the right half of the cell*/
}

int cellj_Glob(const MeshS *pM, const Real y, const Real dx2_1, int *j, Real *b, const int js)
{
  *b = (y - pM->RootMinX[1]) * dx2_1 + (Real)js;
  *j = (int)(*b);
  if (((*b)-(*j)) < 0.5) return 0;	/* in the left half of the cell*/
  else return 1;			/* in the right half of the cell*/
}

int cellk_Glob(const MeshS *pM, const Real z, const Real dx3_1, int *k, Real *c, const int ks)
{
  *c = (z - pM->RootMinX[2]) * dx3_1 + (Real)ks;
  *k = (int)(*c);
  if (((*c)-(*k)) < 0.5) return 0;	/* in the left half of the cell*/
  else return 1;			/* in the right half of the cell*/
}




#include <math.h>
//#include "nrutil.h"
#define ERRTOL 0.08
#define TINY 1.5e-38
#define BIG 3.0e37
#define THIRD (1.0/3.0)
#define C1 (1.0/24.0)
#define C2 0.1
#define C3 (3.0/44.0)
#define C4 (1.0/14.0)

float rf(float x, float y, float z)
  /* Special Functions */
  /* Computes Carlson’S elliptic integral of the first */
  /* kind, RF (x, y, z). x, y, and z must be nonneg- ative,
    and at most one can be zero. */
  /* TINY must be at least 5 times the machine underflow */
  /* limit, BIG at most one fifth the machine overflow limit. */
{

  float alamb,ave,delx,dely,delz,e2,e3,sqrtx,sqrty,sqrtz,xt,yt,zt;

 if (fmin(fmin(x,y),z) < 0.0 || fmin(fmin(x+y,x+z),y+z) < TINY
     || fmax(fmax(x,y),z) > BIG)

   ath_error("invalid arguments in rf"); xt=x;
   yt=y;
   zt=z;

 do {
   sqrtx=sqrt(xt);
   sqrty=sqrt(yt);
   sqrtz=sqrt(zt);
   alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz;
   xt=0.25*(xt+alamb);
   yt=0.25*(yt+alamb);
   zt=0.25*(zt+alamb);
   ave=THIRD*(xt+yt+zt);
   delx=(ave-xt)/ave;
   dely=(ave-yt)/ave;
   delz=(ave-zt)/ave;
 }

 while (fmax(fmax(fabs(delx),fabs(dely)),fabs(delz)) > ERRTOL);
 e2=delx*dely-delz*delz;
 e3=delx*dely*delz;
 return (1.0+(C1*e2-C2-C3*e3)*e2+C4*e3)/sqrt(ave);
}

#undef ERRTOL 
#undef TINY 
#undef BIG 
#undef THIRD 
#undef C1 
#undef C2
#undef C3
#undef C4


#include <math.h>
//#include "nrutil.h"
#define ERRTOL 0.05
#define TINY 1.0e-25
#define BIG 4.5e21
#define C1 (3.0/14.0)
#define C2 (1.0/6.0)
#define C3 (9.0/22.0)
#define C4 (3.0/26.0)
#define C5 (0.25*C3)
#define C6 (1.5*C4)

float rd(float x, float y, float z)
  /* Computes Carlson’s elliptic integral of the second kind, RD(x,y,z). */
  /* x and y must be non- negative, and at most one can be zero. z */
  /* must be positive. TINY must be at least twice the negative 2/3 power */
  /* of the machine overflow limit. BIG must be at most 0.1 × ERRTOL */
  /* times the negative 2/3 power of the machine underflow limit. */
{
  float alamb,ave,delx,dely,delz,ea,eb,ec,ed,ee,fac,sqrtx,sqrty,
    sqrtz,sum,xt,yt,zt;
  if (fmin(x,y) < 0.0 || fmin(x+y,z) < TINY || fmax(fmax(x,y),z) > BIG)
    ath_error("invalid arguments in rd");
  xt=x;
  yt=y;
  zt=z;
  sum=0.0;
  fac=1.0;
  do {
    sqrtx=sqrt(xt);
    sqrty=sqrt(yt);
    sqrtz=sqrt(zt);
    alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz;
    sum += fac/(sqrtz*(zt+alamb)); fac=0.25*fac;
    xt=0.25*(xt+alamb);
    yt=0.25*(yt+alamb);
    zt=0.25*(zt+alamb);
    ave=0.2*(xt+yt+3.0*zt);
    delx=(ave-xt)/ave;
    dely=(ave-yt)/ave;
    delz=(ave-zt)/ave;
  } while (fmax(fmax(fabs(delx),fabs(dely)),fabs(delz)) > ERRTOL);
  ea=delx*dely;
  eb=delz*delz;
  ec=ea-eb;
  ed=ea-6.0*eb;
  ee=ed+ec+ec;
  return 3.0*sum+fac*(1.0+ed*(-C1+C5*ed-C6*delz*ee)
		      +delz*(C2*ee+delz*(-C3*ec+delz*C4*ea)))/(ave*sqrt(ave));
}


float ellf(float phi, float ak)
/* Legendre elliptic integral of the 1st kind F (φ, k), */
/*   evaluated using Carlson’s function RF . */
/*   The argument ranges are 0 ≤ φ ≤ π/2, 0 ≤ ksinφ ≤ 1. */
{
  float rf(float x, float y, float z); float s;
  s=sin(phi);
  return s*rf(SQR(cos(phi)),(1.0-s*ak)*(1.0+s*ak),1.0);
}


/* void Constr_optDepthStackOnGlobGrid(MeshS *pM, GridS *pG){ */
/*   // it is assumed that the source is located on the axis of symmetry */


/*    Real x1is, x2js, x3ks, den,  ri, tj, zk,x1,x2,x3, */
/*    	   	   l,dl,tau=0,dtau=0 ; */
   
/*    int i,j,k,is,ie,js,je,ks,ke,ip,jp,kp,iloc,jloc,kloc, */
/* 	knew,my_id=0,ii; */

/*     Real abs_cart_norm, cart_norm[3], cyl_norm[3], xyz_src[3], xyz_pos[3], rtz_pos[3],rtz_in[3], xyz_p[3], */
/*       res[1], olpos[3],xyz_cc[3], rtz_cc[3]; */
/*     int ijk_cur[3],iter,iter_max,lr,ir=0,i0,j0,k0, */
/*     		NmaxArray=1; */
/*     int nroot; */
/*     Real xyz_in[3], radSrcCyl[3], dist, sint, cost, s2; */
    
/* //    CellOnRayData arrayDataTmp; //indices, ijk and dl */
/* //    int tmpIntArray1, *tmpIntArray2, *tmpIntArray3; */

/*     CellOnRayData *tmpCellIndexAndDisArray; */
/*     Real *tmpRealArray; */

/*     int ncros; */


/*  int  mpi1=1; */

/*  is = 0; */
/*  ie = pM->Nx[0]-1;      */
/*  js = 0; */
/*  je = pM->Nx[1]-1;        */
/*  ks = 0; */
/*  ke = pM->Nx[2]-1; */

  
/*  iter_max= sqrt(pow(ie,2) + pow(je,2) +pow(ke,2)); */

/* //  allocating temporary array for 1D ray from point source */
/* //  CellIndexAndCoords */

/*   if ((tmpCellIndexAndDisArray=(CellOnRayData*)calloc(iter_max , sizeof(CellIndexAndCoords)))== NULL) { */
/*       ath_error("[calloc_1d] failed to allocate memory CellIndexAndCoords\n"); */
/*     } */

/*   cc_posGlobFromGlobIndx(pM, pG,is,js,ks, &x1is,&x2js,&x3ks); */
  
  
/*   for (kp = ks; kp <= ke; kp++) { //z */
/*     for (jp = js; jp <= je; jp++) { //t */
/*       for (ip = is; ip <= ie; ip++) { //r */

/*       #ifndef MPI_PARALLEL /\* if Not parallel *\/ */
/* 	ijkGlobToLoc(pG, ip, jp, kp, &iloc, &jloc, &kloc); */
/* 	pG->yglob[kp][jp][ip].ro = pG->U[kloc][jloc][iloc].d; */
/*       #endif */
	  
/* 	  /\* from ip, jp, kp get rtz position *\/ */
/* 	cc_posGlobFromGlobIndx(pM,pG,ip,jp,kp,&rtz_pos[0],&rtz_pos[1],&rtz_pos[2]); */
	   
/* 	radSrcCyl[0] = 0.; */
/* 	radSrcCyl[1] = rtz_pos[1]; // corresponds to phi_jp */
/* 	radSrcCyl[2] =  0; // a ring at z=0 */

/* // 	   for parallel calls they are not the same with radSrcCyl */
/* 	rtz_in[0] = x1is; */
/* 	rtz_in[1] = rtz_pos[1]; // corresponds to phi_jp */
/* 	rtz_in[2] = rtz_pos[2] * rtz_in[0] / rtz_pos[0]; */

/* 	/\* Cart. position of the entry point *\/ */
/* 	coordCylToCart(xyz_in, rtz_in, cos( rtz_in[1]), sin( rtz_in[1])); */
	   	     
/* 	/\* Cart. position of the source point *\/ */
/* 	coordCylToCart(xyz_src, radSrcCyl, cos(radSrcCyl[1]), sin(radSrcCyl[1])); */
	    
/* 	/\* pG->disp[kp][jp][ip] =  sqrt(pow(rtz_in[0]-radSrcCyl[0],2)+ pow( rtz_in[2]  */
/* 	   - radSrcCyl[2],2)); *\/	     */
/* 	/\* Cart. position of the starting point --> displacemnet*\/ */

/* 	/\* indices of the starting point *\/ */
/* 	lr=celli_Glob(pM,rtz_in[0], 1./pG->dx1, &i0, &ir, is); */
/* 	lr=cellj_Glob(pM,rtz_in[1], 1./pG->dx2, &j0, &ir, js); */
/* 	lr=cellk_Glob(pM,rtz_in[2], 1./pG->dx3, &k0, &ir, ks); */

/* 	ijk_cur[0] = i0; */
/* 	ijk_cur[1] = j0; */
/* 	ijk_cur[2] = k0; */

/* // 	   printf("%f %d \n", x1is, i0); getchar(); */
/* //	    corrected location of the source on the grid */
/* //		cc_pos(pG,i0,j0,k0,&rtz_pos[0],&rtz_pos[1],&rtz_pos[2]); */

/* 	sint = sin(rtz_pos[1]); */
/* 	cost = cos(rtz_pos[1]); */
/* 	/\* from rtz position get get Cart position *\/ */
/* 	coordCylToCart(xyz_p, rtz_pos, cost, sint); */

/* 	/\* get normalized  vector from start to finish *\/ */
/* 	for(i=0;i<=2;i++) cart_norm[i]= xyz_p[i]-xyz_src[i]; */

/* 	abs_cart_norm = sqrt(pow(cart_norm[0], 2)+pow(cart_norm[1], 2)+pow(cart_norm[2], 2)); */

/* 	dist = absValueVector(cart_norm); */
	
/* 	for(i=0;i<=2;i++) cart_norm[i] = cart_norm[i]/abs_cart_norm; */
/* 	cost = cos(radSrcCyl[1]); */
/* 	sint = sin(radSrcCyl[1]); */
/* 	cartVectorToCylVector(cyl_norm, cart_norm, cos(radSrcCyl[1]), sin(radSrcCyl[1])); */
/* 	/\* tau = pG->tau_e[ kp ][ jp ][ 1 ]; *\/ */
/* 	/\* pG->tau_e[kp][jp] [ is ] = tau; *\/ */
	
/* 	/\* starting point for ray-tracing *\/ */
/* 	rtz_pos[0]=rtz_in[0]; */
/* 	rtz_pos[1]=rtz_in[1]; */
/* 	rtz_pos[2]=rtz_in[2]; */
/* 	sint = sin(rtz_pos[1]); */
/* 	cost = cos(rtz_pos[1]); */
  
/* 	for(i=0;i<=2;i++){ */
/* 	  xyz_pos[i]=xyz_in[i]; */
/* 	  xyz_cc[i]=xyz_in[i]; */
/* 	} */
		
/* 	l = 0; */
/* 	dl =0; */
/* 	tau = 0; */
/* 	dtau=0; */

/* 	for (iter=0; iter<=iter_max; iter++){ */

/* 	  s2 = pow(xyz_p[0]-xyz_pos[0],2)+pow(xyz_p[1]-xyz_pos[1],2)+pow(xyz_p[2]-xyz_pos[2],2); */
/* 	  olpos[0]=xyz_cc[0]; */
/* 	  olpos[1]=xyz_cc[1]; */
/* 	  olpos[2]=xyz_cc[2]; */
		  
/* 	  traceGridCellOnGlobGrid(pM, pG, res, ijk_cur, xyz_pos, rtz_pos, */
/* 				  cart_norm, cyl_norm, &ncros); */
/* 	  /\* find new cyl coordinates: *\/ */
/* 	  cc_posGlobFromGlobIndx(pM,pG,ijk_cur[0],ijk_cur[1],ijk_cur[2], */
/* 				 &rtz_pos[0],&rtz_pos[1],&rtz_pos[2]); */

/* 	  ri = sqrt(pow(xyz_pos[0],2)+pow(xyz_pos[1],2)); */
/* 	  tj = atan2(xyz_pos[1], xyz_pos[0]); */
/* 	  zk =xyz_pos[2]; */

/* 	  /\* we need only sign(s) of cyl_norm *\/ */
/* 	  cyl_norm[0] = cart_norm[0]*cos(tj) + cart_norm[1]*sin(tj); */
/* 	  cyl_norm[1] = -cart_norm[0]*sin(tj) + cart_norm[1]*cos(tj); */

/* 	  dl = res[0]; */

/* 	  rtz_cc[0]=rtz_pos[0]; */
/* 	  rtz_cc[1]=rtz_pos[1]; */
/* 	  rtz_cc[2]=rtz_pos[2]; */
		  
/* 	  coordCylToCart(xyz_cc, rtz_cc,  cos(rtz_cc[1]),  sin(rtz_cc[1]) ); */
		  
/* 	  dl = sqrt(pow(xyz_cc[0]-olpos[0],2)+pow(xyz_cc[1]-olpos[1],2)+ */
/* 		    pow(xyz_cc[2]-olpos[2],2)); */

/* 	  /\* dl = sqrt( pow(pG->dx1,2)  +  pow(pG->dx3,2) ); *\/ */

/* 	  (tmpCellIndexAndDisArray[iter]).dl = dl; */
/* 	  (tmpCellIndexAndDisArray[iter]).i = ijk_cur[0]; */
/* 	  (tmpCellIndexAndDisArray[iter]).j = ijk_cur[1]; */
/* 	  (tmpCellIndexAndDisArray[iter]).k = ijk_cur[2]; */

/* 	  l += dl; */
	  
/* /\* #ifdef MPI_PARALLEL *\/ */
/* /\* 	  den = (pG->yglob[ijk_cur[2]] [ijk_cur[1]] [ijk_cur[0]].ro < 1.05* rho0) ? *\/ */
/* /\* 	    tiny : pG->yglob[ijk_cur[2]][ijk_cur[1]][ijk_cur[0]].ro; *\/ */
/* /\* #else *\/ */
/* /\* 	  den = (pG->U[ijk_cur[2]][ijk_cur[1]][ijk_cur[0]].d < 1.05* rho0) ? *\/ */
/* /\* 	    tiny : pG->U[ijk_cur[2]][ijk_cur[1]][ijk_cur[0]].d;		  *\/ */
		    
/* /\* #endif *\/ */
		  
/* /\* 	  dtau = dl * den; *\/ */

/* /\* 	  tau  += dtau; *\/ */
		  
/* 	  NmaxArray = iter+1; //To use in allocation */

/* 	  //		  tau = l; */
/* 	  if (pow(xyz_p[0]-xyz_pos[0],2)+pow(xyz_p[1]-xyz_pos[1],2)+ */
/* 	      pow(xyz_p[2]-xyz_pos[2],2)>s2){ */
/* 	    //			tau= 0; */
/* 	    break; */
/* 	  } */
		 

/* 	  /\* test if reached to the ip,jp,kp *\/ */
/* 	  if( (ijk_cur[0]==ip &&  ijk_cur[1]==jp &&  ijk_cur[2]==kp) */
/* 	      /\* || (ijk_cur[]==ip &&  ijk_cur[1]==jp &&  ijk_cur[2]==kp) *\/ */
/* 	      ){ */

/* 	    // cc_pos(pG, ijk_cur[0], ijk_cur[1], ijk_cur[2], &x1, &x2, &x3); */

/* 	    break; */
/* 	  }		   */
/* 	} //iter */

/* 	//pG->GridOfRays should be already allocated at init_grid.c */
/* 	(pG->GridOfRays[kp][jp][ip]).Ray = */
/* 	  (CellOnRayData*)calloc_1d_array(NmaxArray,sizeof(CellOnRayData)); */

/* 	(pG->GridOfRays[kp][jp][ip]).len = NmaxArray; */

/* 	pG->tau_e[kp][jp][ip] = 0; */


/* 	for (ii = 0; ii<NmaxArray; ii++){ */

/* 	  (pG->GridOfRays[kp][jp][ip]).Ray[ii].dl= (tmpCellIndexAndDisArray[ii]).dl; */
/* 	  (pG->GridOfRays[kp][jp][ip]).Ray[ii].i=(tmpCellIndexAndDisArray[ii]).i; */
/* 	  (pG->GridOfRays[kp][jp][ip]).Ray[ii].j=(tmpCellIndexAndDisArray[ii]).j; */
/* 	  (pG->GridOfRays[kp][jp][ip]).Ray[ii].k=(tmpCellIndexAndDisArray[ii]).k; */

/* 	  //		   printf("NmaxArray= %d dl \n\n ", NmaxArray); */
	
/* 	  /\* pG->tau_e[kp][jp][ip] += dl; *\/ */
/* 	} */
	
	
/* 	/\* for (ii = 0; ii<(pG->GridOfRays[kp][jp][ip]).len; ii++){ *\/ */
/* 	/\* 	i=(pG->GridOfRays[kp][jp][ip]).Ray[ii].i; *\/ */
/* 	/\* 	j=(pG->GridOfRays[kp][jp][ip]).Ray[ii].j; *\/ */
/* 	/\* 	k=(pG->GridOfRays[kp][jp][ip]).Ray[ii].k; *\/ */
/* 	/\* 	dl = (pG->GridOfRays[kp][jp][ip].Ray[ii]).dl; *\/ */
			
/* 	/\* 	dtau = dl; *\/ */
/* 	/\* 	tau+=dtau; *\/ */
/* 	/\* } *\/ */

	

		      
		  
/* 	//		pG->tau_e[kp][jp][ip] *= Rsc*Dsc*KPE; */
/* 	//		for (ii = 0; ii<(pG->GridOfRays[kp][jp][ip]).len; ii++){ */
/* 	//			pG->tau_e[kp][jp][ip] += (pG->GridOfRays[kp][jp][ip].Ray[ii]).dl; */
/* 	//		} */

/* 	//pG->yglob[kp][jp][ip].tau  =  tau; //pG->tau_e[kp][jp][ip]; */

/* 	tau=0; */


/*       } //ip */

      
/*     } //jp */
/*   } //kp */

/*   /\* for (j = js; j <=je; j++) printf("%f \n",  pG->yglob[ks][j][is].tau  ); *\/ */


/* free(tmpCellIndexAndDisArray); */

/* //  printf(" optDepthStack \n"); */
/* //  getchar(); */
/* } */


	/* Side = CheckCrossBlockBdry(i, k, cur_id); */
	       
	  /* if(my_id==1) printf("my_id %d, cur_id %d \n", my_id, cur_id); */
	  
	  /* tmpRaySegGridPerBlock[ib]  */
	   
	    
	    
	
      /* 	If((Side==LS)||(Side==RS)||(Side==DS)||(Side==US)){ */
      /* 	  switch (InOutBlock){     */
      /* 	  case (OutBlock): */
	    
      /* 	    ii_s = ii;      */

      /* 	    /\* tmpRaySegGridPerBlock[SegId].EnterSide=Side; *\/ */
	    
      /* 	    InOutBlock = InBlock; */
      /* 	    NumOfSegInBlock +=1;	     */
      /* 	    break; */
      /* 	  case (InBlock):   */
      /* 	    ii_e = ii; */

      /* 	    /\* tmpRaySegGridPerBlock[SegId].ExitSide=Side; *\/ */
	    
      /* 	    break; */
      /* 	  default:       */
      /* 	    ath_error("Error in Constr_SegmentsFromRays"); */
      /* 	  } */
      /* 	} */
      /* 	Side = NotOnSide;	 */
      /* } /\* ii-ray *\/ */
      
      /* if (InOutBlock == InBlock){ */
      /* 	tmpRaySegGridPerBlock.id[0]=ig; */
      /* 	tmpRaySegGridPerBlock.id[1]=kg;	   */
      /* } */


/* void testSegments2_dump(GridS *pG, int my_id, int TestType){ */

/*   int ig_s = BufBndr[my_id][0]; */
/*   int ig_e = BufBndr[my_id][3]; */
/*   int kg_s = BufBndr[my_id][2]; */
/*   int kg_e = BufBndr[my_id][5]; */


/*   int js =  BufBndr[my_id][1];     */
/*   int je = BufBndr[my_id][4] ; */
/*   int phi_sz = je-js+1; */
  
/*    int mpi1=1; */
/*    while(mpi1==1); */

  
  
  
/*   int dlen; */
/*   /\* int l = pG->BlockToRaySegMask[kp][ip]; *\/ */
      
/*   switch (TestType){ */
/*   case 1:    */
/*     //printf("id: %d, (l,%d), %d, %d, \n", my_id, l, kp, ip); */
/*     break; */
/*   case 2: */
   
/*       /\* calc test data on all segs/Block for testing*\/ */
/*     { */
/*     Real dat=0; */

/*       for(int l=0; l <= NumSegBlock_glb-1; l++){ */

/* 	/\* if(( pG->RaySegGridPerBlock[l].dat_vec_1d= *\/ */
/*       	/\*   (float*)calloc_1d_array(phi_sz,sizeof(float)))==NULL) *\/ */
/*       	/\* ath_error("[calloc_1d] failed to allocate (pG->RaySegGridPerBlock)[l].d_tau \n"); *\/ */

/* 	/\* if ( (pG->RaySegGridPerBlock)[l].dat_vec_1d_size>0){ *\/ */
/* 	/\*   for(int m=0; m <= (pG->RaySegGridPerBlock)[l].data_len-1; m++){      *\/ */
/* 	/\*     dat += (pG->RaySegGridPerBlock)[l].data[m].dl;	   *\/ */
/* 	/\*   } *\/ */
/* 	/\* } *\/ */

/* 	//	printf("here  %d %d %d \n", my_id, l, pG->RaySegGridPerBlock[l].dat_vec_1d_size); */

/* 	if ( (pG->RaySegGridPerBlock)[l].dat_vec_1d_size>0){ */
	  
/* 	  for (int j = js; j<= phi_sz-1; j++) { */

/* 	    //  pG->RaySegGridPerBlock[l].dat_vec_1d[j] = dat; */

/* 	  } */
/* 	} */

	
/*       } */


/*       //1) */
/*       if( (send_1d_buf = (float*)calloc_1d_array(phi_sz, sizeof(float)))==NULL) */
/*         ath_error("[calloc_1d] failed to allocate send_1d_buf \n"); */

/*       if( (recv_1d_buf = (float*)calloc_1d_array(phi_sz, sizeof(float)))==NULL) */
/*         ath_error("[calloc_1d] failed to allocate send_1d_buf \n"); */

     
/*     } */
	
/*     {//create MPI datatype */
            
/*       const int nitems=3; */
/*       int blocklengths[3] = {1,1,1};	   */
/*       MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_FLOAT}; */
/*       MPI_Datatype mpi_seg_data;	  */
/*       MPI_Aint offsets[3]; */

/*       offsets[0] = offsetof(SegHeadId, i); */
/*       offsets[1] = offsetof(SegHeadId, k); */
/*       offsets[2] = offsetof(SegHeadId, dat);	   */

/*       MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_seg_data); */
/*       MPI_Type_commit(&mpi_seg_data); */
		
/*       SegHeadId SegSendBuf, SegRecvBuf; */
	  
/*       const int tag = 13, tag1d=11; */

/*       /\* for (int l=0; l <= NumSegBlock_glb - 1; l++){	 *\/ */

/* 	int dest_id = pG->RaySegGridPerBlock[l].mpi_id_ik_head_block[0];	 */

/* 	int i_loc, k_loc, l_loc; */
  
/* 	if (my_id != dest_id){ */

/* 	  SegSendBuf.i = pG->RaySegGridPerBlock[l].mpi_id_ik_head_block[1]; */
/* 	  SegSendBuf.k = pG->RaySegGridPerBlock[l].mpi_id_ik_head_block[2];	   */
/* 	  SegSendBuf.dat = pG->RaySegGridPerBlock[l].dTau; */
/* 	  if(pG->RaySegGridPerBlock[l].type == Head) ath_error("error in testSegments2");	     */
/* 	  MPI_Send(&SegSendBuf, 1, mpi_seg_data, dest_id, tag, MPI_COMM_WORLD); */

/* 	  /\* send the phi-dependent data *\/	   */
/* 	  /\* MPI_Send(&pG->RaySegGridPerBlock[l].dat_vec_1d[0], *\/ */
/* 	  /\* 	   phi_sz, MPI_DOUBLE, dest_id, tag1d, MPI_COMM_WORLD); *\/ */


/* 	  if (pG->RaySegGridPerBlock[l].type == Upstr){ */

/* 	    for(int m = 0; m <= phi_sz-1; m++) */
/* 	      send_1d_buf[m] = pG->RaySegGridPerBlock[l].dat_vec_1d[m];	   */

/* 	    printf(" --- sending %d %d %d \n", my_id, phi_sz, dest_id);	   */

/* 	    MPI_Send(&send_1d_buf[0], phi_sz, MPI_FLOAT, dest_id, tag1d, MPI_COMM_WORLD); */

/* 	  } */
/* 	}   */
	  

/* 	if (my_id == dest_id){ */
	    
/* 	  MPI_Status status; */
/* 	  /\* we dont need id of my_id, i.e. relevant size = real_size-2 *\/	     */
/* 	  int con_arr_sz =  (pG->RaySegGridPerBlock)[l].MPI_idConnectArray_size; */
	    
/* 	  for(int impi=0; impi <= con_arr_sz-2; impi++){ */
/* 	    /\* iterate over upstream segments, current elem not included *\/ */

/* 	    int orig_id = pG->RaySegGridPerBlock[l].MPI_idConnectArray[impi]; */
/* 	    //printf("MPI connect, my_id=%d, mpi_indx=%d \n", my_id,mpi_indx); */

/* 	    /\* check if there is smth to receive *\/ */
/* 	    MPI_Recv(&SegRecvBuf, 1, mpi_seg_data, orig_id, tag, MPI_COMM_WORLD, &status);	       */

/* 	    /\* printf("from:%d to:%d, %d, %d %f \n", orig_id, dest_id, SegRecvBuf.i, *\/ */
/* 	    /\* 	     SegRecvBuf.k, SegRecvBuf.dat); *\/ */
/* 	    k_loc=SegRecvBuf.k; */
/* 	    i_loc=SegRecvBuf.i;	       */
/* 	    l_loc=pG->BlockToRaySegMask[k_loc][i_loc]; */
/* 	    /\* printf("id = %d, lloc= %d \n", my_id, l_loc); *\/		 */
/* 	    pG->RaySegGridPerBlock[l_loc].dTau += SegRecvBuf.dat; */
	   
	  

/* 	    MPI_Recv(&recv_1d_buf, phi_sz, MPI_FLOAT, orig_id, tag1d, */
/* 	    	     MPI_COMM_WORLD, &status); */

/* 	      /\* } *\/ */
	     
/* 	    /\* printf(" ++++  recv %d %d %d\n", my_id, phi_sz, orig_id); *\/	     */
/* 	    /\* receive the segment angle dependent data *\/ */

/* 	    //if ( (pG->RaySegGridPerBlock)[l].dat_vec_1d_size>0){ */

/* 	    if (con_arr_sz >1 ){ /\* avoid blocks closest to the source *\/ */

/* 	      //for(int m = 0; m <= phi_sz-1; m++) { */

/* 	      pG->RaySegGridPerBlock[l_loc].dat_vec_1d[0] =1.; */
	      
/*   //		pG->RaySegGridPerBlock[l_loc].dat_vec_1d[m] = recv_1d_buf[m]; */
	      
/* 		//} */
/* 	    } */
	  
	    
/* 	  } */
	    
/* 	  if (con_arr_sz <= 1){//closest to source	       */
/* 	    i_loc=pG->RaySegGridPerBlock[l].mpi_id_ik_head_block[1]; */
/* 	    k_loc=pG->RaySegGridPerBlock[l].mpi_id_ik_head_block[2]; */
/* 	    l_loc=pG->BlockToRaySegMask[k_loc][i_loc]; */
	       
/* 	  } */

	    
/* 	  /\* calc the same along the Ray *\/ */
/* 	  Real s=0; */
/* 	  int len=(pG->GridOfRays[k_loc][i_loc]).len; */

/* 	  for (int ii = 0; ii<len; ii++){ */
/* 	    s += (pG->GridOfRays[k_loc][i_loc]).Ray[ii].dl; */
/* 	    //	      	printf("%d, %f \n", ii, s); */
/* 	  } */

/* 	  printf("id:%d, s=%f, dat=%f \n",my_id, s, pG->RaySegGridPerBlock[l_loc].dTau); */
	    	    
/* 	} /\* my_id == dest_id *\/ */
	  	  
/*       }	  	 */
/*     }/\* end seg-loop *\/ */

/*     break; */
	
/*   default: */
/*     printf("%d \n", TestType); */
/*     ath_error("unknown case in testSegments2"); */
/*   } // switch           */
/* } */
