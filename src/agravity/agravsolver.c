
#include <math.h>
#include <stdio.h>

#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../defs.h"
#include "../athena.h"
#include "../globals.h"
#include "../prototypes.h"

// #include "aprototypes.h"

#ifdef sgHYPRE
/* =================================================================================== */
/*                                   GRAVITY SOLVER                                    */
/* =================================================================================== */




#include "fftw3.h"

#include "HYPRE_struct_ls.h"

void ijkGlobToLoc(const GridS *pG, const int iglob, const int jglob, 
            const int kglob, int *iloc, int *jloc, int *kloc);
void ijkLocToGlob(const GridS *pG, const int iloc, const int jloc,const int kloc,
		  int *iglob, int *jglob, int *kglob);



#define REAL 0
#define IMAG 1

#ifdef FFT_ENABLED
#define GNUPLOT "gnuplot -persist"

void  testFourier(MeshS *pM, GridS *pG){
int i,j,k,is,ie,js,je,ks,ke, 
    ip,jp,kp,Nfft;
Real xi,tj,zk,t;


fftw_complex *sig, *sigBack, *out;
fftw_plan pln, plnBack;

  is = pG->is;
  ie = pG->ie;
  js = pG->js;
  je = pG->je;
  ks = pG->ks;
  ke = pG->ke;
  
  Nfft = je;
  
  // clock_t time;  
  // time = clock();
  // calcFFT();  

  // signal 
  sig  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nfft);
  // result
  out  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nfft);
  pln = fftw_plan_dft_1d(Nfft, sig, out, FFTW_FORWARD, FFTW_ESTIMATE);
  sigBack = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nfft);
  plnBack = fftw_plan_dft_1d(Nfft, out, sigBack, FFTW_FORWARD, FFTW_ESTIMATE);

  kp = 10;
  ip = 10;
  
  // apause();

  Real dt = 1./Nfft;
  Real f0 = 40;

  for (j = 0; j<Nfft; j++){
    
    t = (double)j / (double)Nfft;    

    sig[j][IMAG] = 0;
    sig[j][REAL] = 0;
    sig[j][REAL] = sin(f0*2.0*M_PI*t);

    // sig[j][REAL] = 1.0 * cos(2*M_PI* Om * t)+cos(2*M_PI*2* Om * t);
    // cc_pos(pG,ip, j, kp, &xi, &tj, &zk);

  }

   fftw_execute(pln); /* repeat as needed */
   fftw_destroy_plan(pln);      
   fftw_execute(plnBack);

    FILE *gp;
    gp = popen(GNUPLOT,"w"); /* 'gp' is the pipe descriptor */
    if (gp==NULL)
        {
          printf("Error opening pipe to GNU plot. Check if you have it! \n");
          exit(0);
        }
    fprintf(gp, "plot '-' with lines\n");
   
    for (i=0; i< div(Nfft,2).quot; i++){

      Real res = sqrt(pow(out[i][REAL],2) + pow(out[i][IMAG],2));
      
      // res = sqrt(pow(sigBack[i][REAL],2) + pow(sigBack[i][IMAG],2))/Nfft;


      // printf( "%d out: %g \n", i, res);         
      
      fprintf(gp, "%g\n", res);

      // fprintf(gp, "%g\n", (double)(i) );
      };
    fflush(gp); 
    fprintf(gp, "e\n"); 
    fclose(gp);

//  fprintf(gp, "plot '-' with lines\n");
/*   fc_pos(pG,is,js,ks, &x1is,&x2js,&x3ks); */
    
   
    fftw_free(sig);
    fftw_free(out);
    // time = clock() - t;
    // double time_taken = ((double)time)/CLOCKS_PER_SEC; // in seconds
    // printf("took %f seconds to execute \n", time_taken);

printf(" ................ done .......... \n");

}

#endif


// void calcGravitySolverParams(MeshS *pM, GridS *pG){
//   Real res;
//   res = rf(0.1, 0.1, 0.1);
//   printf("calcGravitySolverParams %f \n", res);
//   apause();
// }

// void gravPotAtBndry(MeshS *pM, GridS *pG){
//   /* calculate grav pot at the boundary */
// }


#define REAL 0
#define IMAG 1
#define Nfft 128

void calcFFT(){

  // fftw_complex *in, *out;
  fftw_complex *sig, *sig2, *out;

  fftw_plan pln, planBack;
  
  int i;
  float theta, res, om,arg;
  
  om=15;

// signal 
  sig  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nfft);

// result
  out  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nfft);
  
  pln = fftw_plan_dft_1d(Nfft, sig, out, FFTW_FORWARD, FFTW_ESTIMATE);

  sig2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nfft);
  planBack = fftw_plan_dft_1d(Nfft, out, sig2, FFTW_FORWARD, FFTW_ESTIMATE);


  
  for (i = 0; i < Nfft; i++) {     
    theta = (double)i / (double)Nfft * M_PI;    
    sig[i][REAL] = 1.0 * cos(om * theta) + 0.5 * cos(25.0 * theta);
    sig[i][IMAG] = 1.0 * sin(-om * theta) + 0.5 * sin(25.0 * theta);    
    printf("theta=  %f %f %f \n", theta, sig[i][REAL], sig[i][IMAG]);
  }

  
  fftw_execute(pln); /* repeat as needed */

  for (i = 0; i < Nfft; i++) {
     printf("freq: %3d %+9.5f %+9.5f I\n", i, out[i][REAL], out[i][IMAG]);
  }
 
   
  fftw_destroy_plan(pln);


  printf("\nInverse transform:\n");
  planBack = fftw_plan_dft_1d(Nfft, out, sig2, FFTW_BACKWARD, FFTW_ESTIMATE);
  
  fftw_execute(planBack);

  // for (i = 0; i < Nfft; i++){
  //   printf("recover: %3d %+9.5f %+9.5f I vs. %+9.5f %+9.5f I\n",
	//    i, sig1[i][0], in[i][1], in2[i][0], in2[i][1]);
  // }
  
  fftw_destroy_plan(planBack);
  
  fftw_free(sig);
  fftw_free(out);

  printf("calcFFT done ..\n\n");

  // getchar();

}

// #ifdef MPI_PARALLEL

void testHYPRE_ex2( MeshS *pM, GridS *pG){
  int i,j,k,is,ie,js,je,ks,ke,ip,jp,kp,i_g, j_g, k_g;
  int ng1, ng2;
  

   int myid, num_procs;

   int vis = 0;

   HYPRE_StructGrid     grid;
   HYPRE_StructStencil  stencil;
   HYPRE_StructMatrix   A;
   HYPRE_StructVector   b;
   HYPRE_StructVector   x;
   HYPRE_StructSolver   solver;
   HYPRE_StructSolver   precond;

   /* Initialize MPI */
   /* MPI_Init(&argc, &argv); */
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   if (num_procs != 2)
   {
      if (myid == 0) printf("Must run with 2 processors!\n");
      apause();
      /* MPI_Finalize(); */
    
   }
 
   /* 1. Set up a grid */
   {
      /* Create an empty 2D grid object */
      HYPRE_StructGridCreate(MPI_COMM_WORLD, 2, &grid);

      /* Processor 0 owns two boxes in the grid. */
      if (myid == 0)
      {
         /* Add a new box to the grid */
         {
            int ilower[2] = {-3, 1};
            int iupper[2] = {-1, 2};

            HYPRE_StructGridSetExtents(grid, ilower, iupper);
         }

         /* Add a new box to the grid */
         {
            int ilower[2] = {0, 1};
            int iupper[2] = {2, 4};

            HYPRE_StructGridSetExtents(grid, ilower, iupper);
         }
      }

      /* Processor 1 owns one box in the grid. */
      else if (myid == 1)
      {
         /* Add a new box to the grid */
         {
            int ilower[2] = {3, 1};
            int iupper[2] = {6, 4};

            HYPRE_StructGridSetExtents(grid, ilower, iupper);
         }
      }

      /* This is a collective call finalizing the grid assembly.
         The grid is now ``ready to be used'' */
      HYPRE_StructGridAssemble(grid);
   }

   /* 2. Define the discretization stencil */
   {
      /* Create an empty 2D, 5-pt stencil object */
      HYPRE_StructStencilCreate(2, 5, &stencil);

      /* Define the geometry of the stencil. Each represents a
         relative offset (in the index space). */
      {
         int entry;
         int offsets[5][2] = {{0,0}, {-1,0}, {1,0}, {0,-1}, {0,1}};

         /* Assign each of the 5 stencil entries */
         for (entry = 0; entry < 5; entry++)
            HYPRE_StructStencilSetElement(stencil, entry, offsets[entry]);
      }
   }

   /* 3. Set up a Struct Matrix */
   {
      /* Create an empty matrix object */
      HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &A);

      /* Indicate that the matrix coefficients are ready to be set */
      HYPRE_StructMatrixInitialize(A);

      if (myid == 0)
      {
         /* Set the matrix coefficients for some set of stencil entries
            over all the gridpoints in my first box (account for boundary
            grid points later) */
         {
            int ilower[2] = {-3, 1};
            int iupper[2] = {-1, 2};

            int nentries = 5;
            int nvalues  = 30; /* 6 grid points, each with 5 stencil entries */
            double values[30];

            int stencil_indices[5];
            for (j = 0; j < nentries; j++) /* label the stencil indices -
                                              these correspond to the offsets
                                              defined above */
               stencil_indices[j] = j;

            for (i = 0; i < nvalues; i += nentries)
            {
               values[i] = 4.0;
               for (j = 1; j < nentries; j++)
                  values[i+j] = -1.0;
            }

            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, nentries,
                                           stencil_indices, values);
         }

         /* Set the matrix coefficients for some set of stencil entries
            over the gridpoints in my second box */
         {
            int ilower[2] = {0, 1};
            int iupper[2] = {2, 4};

            int nentries = 5;
            int nvalues  = 60; /* 12 grid points, each with 5 stencil entries */
            double values[60];

            int stencil_indices[5];
            for (j = 0; j < nentries; j++)
               stencil_indices[j] = j;

            for (i = 0; i < nvalues; i += nentries)
            {
               values[i] = 4.0;
               for (j = 1; j < nentries; j++)
                  values[i+j] = -1.0;
            }

            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, nentries,
                                           stencil_indices, values);
         }
      }
      else if (myid == 1)
      {
         /* Set the matrix coefficients for some set of stencil entries
            over the gridpoints in my box */
         {
            int ilower[2] = {3, 1};
            int iupper[2] = {6, 4};

            int nentries = 5;
            int nvalues  = 80; /* 16 grid points, each with 5 stencil entries */
            double values[80];

            int stencil_indices[5];
            for (j = 0; j < nentries; j++)
               stencil_indices[j] = j;

            for (i = 0; i < nvalues; i += nentries)
            {
               values[i] = 4.0;
               for (j = 1; j < nentries; j++)
                  values[i+j] = -1.0;
            }

            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, nentries,
                                           stencil_indices, values);
         }
      }

      /* For each box, set any coefficients that reach ouside of the
         boundary to 0 */
      if (myid == 0)
      {
         int maxnvalues = 6;
         double values[6];

         for (i = 0; i < maxnvalues; i++)
            values[i] = 0.0;

         {
            /* Values below our first AND second box */
            int ilower[2] = {-3, 1};
            int iupper[2] = { 2, 1};

            int stencil_indices[1] = {3};

            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
         }

         {
            /* Values to the left of our first box */
            int ilower[2] = {-3, 1};
            int iupper[2] = {-3, 2};

            int stencil_indices[1] = {1};

            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
         }

         /* { */
         /*    /\* Values above our first box *\/ */
         /*    int ilower[2] = {-3, 2}; */
         /*    int iupper[2] = {-1, 2}; */

         /*    int stencil_indices[1] = {4}; */

         /*    HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1, */
         /*                                   stencil_indices, values); */
         /* } */
	 

         {
            /* Values to the left of our second box (that do not border the
               first box). */
            int ilower[2] = { 0, 3};
            int iupper[2] = { 0, 4};

            int stencil_indices[1] = {1};

            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
         }

         {
            /* Values above our second box */
            int ilower[2] = { 0, 4};
            int iupper[2] = { 2, 4};

            int stencil_indices[1] = {4};

            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
         }
      }
      else if (myid == 1)
      {
         int maxnvalues = 4;
         double values[4];
         for (i = 0; i < maxnvalues; i++)
            values[i] = 0.0;

         {
            /* Values below our box */
            int ilower[2] = { 3, 1};
            int iupper[2] = { 6, 1};

            int stencil_indices[1] = {3};

            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
         }

         {
            /* Values to the right of our box */
            int ilower[2] = { 6, 1};
            int iupper[2] = { 6, 4};

            int stencil_indices[1] = {2};

            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
         }

         {
            /* Values above our box */
            int ilower[2] = { 3, 4};
            int iupper[2] = { 6, 4};

            int stencil_indices[1] = {4};

            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
         }
      }

      /* This is a collective call finalizing the matrix assembly.
         The matrix is now ``ready to be used'' */
      HYPRE_StructMatrixAssemble(A);
   }

   /* 4. Set up Struct Vectors for b and x */
   {
      /* Create an empty vector object */
      HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &b);
      HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &x);

      /* Indicate that the vector coefficients are ready to be set */
      HYPRE_StructVectorInitialize(b);
      HYPRE_StructVectorInitialize(x);

      if (myid == 0)
      {
         /* Set the vector coefficients over the gridpoints in my first box */
         {
            int ilower[2] = {-3, 1};
            int iupper[2] = {-1, 2};

            int nvalues = 6;  /* 6 grid points */
            double values[6];

            for (i = 0; i < nvalues; i ++)
               values[i] = 1.0;
            HYPRE_StructVectorSetBoxValues(b, ilower, iupper, values);

            for (i = 0; i < nvalues; i ++)
               values[i] = 0.0;
            HYPRE_StructVectorSetBoxValues(x, ilower, iupper, values);
         }

         /* Set the vector coefficients over the gridpoints in my second box */
         {
            int ilower[2] = { 0, 1};
            int iupper[2] = { 2, 4};

            int nvalues = 12; /* 12 grid points */
            double values[12];

            for (i = 0; i < nvalues; i ++)
               values[i] = 1.0;
            HYPRE_StructVectorSetBoxValues(b, ilower, iupper, values);

            for (i = 0; i < nvalues; i ++)
               values[i] = 0.0;
            HYPRE_StructVectorSetBoxValues(x, ilower, iupper, values);
         }
      }
      else if (myid == 1)
      {
         /* Set the vector coefficients over the gridpoints in my box */
         {
            int ilower[2] = { 3, 1};
            int iupper[2] = { 6, 4};

            int nvalues = 16; /* 16 grid points */
            double values[16];

            for (i = 0; i < nvalues; i ++)
               values[i] = 1.0;
            HYPRE_StructVectorSetBoxValues(b, ilower, iupper, values);

            for (i = 0; i < nvalues; i ++)
               values[i] = 0.0;
            HYPRE_StructVectorSetBoxValues(x, ilower, iupper, values);
         }
      }

      /* This is a collective call finalizing the vector assembly.
         The vectors are now ``ready to be used'' */
      HYPRE_StructVectorAssemble(b);
      HYPRE_StructVectorAssemble(x);
   }


   /* 5. Set up and use a solver (See the Reference Manual for descriptions
      of all of the options.) */
   {
      /* Create an empty PCG Struct solver */
      HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);

      /* Set PCG parameters */
      HYPRE_StructPCGSetTol(solver, 1.0e-06);
      HYPRE_StructPCGSetPrintLevel(solver, 2);
      HYPRE_StructPCGSetMaxIter(solver, 50);

      /* Use symmetric SMG as preconditioner */
      HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
      HYPRE_StructSMGSetMaxIter(precond, 1);
      HYPRE_StructSMGSetTol(precond, 0.0);
      HYPRE_StructSMGSetZeroGuess(precond);
      HYPRE_StructSMGSetNumPreRelax(precond, 1);
      HYPRE_StructSMGSetNumPostRelax(precond, 1);

      /* Set preconditioner and solve */
      HYPRE_StructPCGSetPrecond(solver, HYPRE_StructSMGSolve,
                                HYPRE_StructSMGSetup, precond);
      HYPRE_StructPCGSetup(solver, A, b, x);
      HYPRE_StructPCGSolve(solver, A, b, x);
   }


  
}


void testHYPRE_1( MeshS *pM, GridS *pG){
  int NumDomsInGlob;

  int i,j,k,is,ie,js,je,ks,ke,ip,jp,kp,i_g, j_g, k_g;
  int ng1, ng2;
  
  HYPRE_StructGrid     grid;
  HYPRE_StructStencil  stencil;
  HYPRE_StructMatrix   A;
  HYPRE_StructVector   b;
  HYPRE_StructVector   x;
  HYPRE_StructSolver   solver;
  HYPRE_StructSolver   precond;
   

  // MPI_Comm_size(MPI_COMM_WORLD, &NumDomsInGlob);
  // if (NumDomsInGlob<2) {
  //   ath_error("testHYPRE_1 should be run with MPI");
  // }
    
  is = pG->is;
  ie = pG->ie;
  js = pG->js;
  je = pG->je;
  ks = pG->ks;
  ke = pG->ke;

  int my_id;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

  int mpi1=1;
  while(mpi1 == 1 );


  double h2 = pG->dx1 * pG->dx2;
  
  int num_iterations;
  double final_res_norm;
  int  ig_s,jg_s,kg_s,ig_e,jg_e,kg_e;

  ijkLocToGlob(pG,is,js,ks, &ig_s, &jg_s, &kg_s);
  ijkLocToGlob(pG,ie,je,ke, &ig_e, &jg_e, &kg_e);

  /* ig_s = 0; */
  /* kg_s = 0; */
  
  /* ig_e = 9; */
  /* kg_e = 9; */
  
  int ilower[2] = {ig_s, kg_s}; 
  int iupper[2] = {ig_e, kg_e};

  ng1 = iupper[0]- ilower[0] + 1; // size of the loc grid 
  ng2 = iupper[1]- ilower[1] + 1;
  
   /* 1. Set up a grid */
   {
     /* Figure out the extents of each processor's piece of the grid. */

      //  this can be done at the beginning 
            
      /* Create an empty 2D grid object */     

      HYPRE_StructGridCreate(MPI_COMM_WORLD, 2, &grid);

      /* Add a new box to the grid */
      HYPRE_StructGridSetExtents(grid, ilower, iupper);

      /* This is a collective call finalizing the grid assembly.
         The grid is now ``ready to be used'' */
      HYPRE_StructGridAssemble(grid);
   }

  /* 2. Define the discretization stencil */
   {
      /* Create an empty 2D, 5-pt stencil object */
      HYPRE_StructStencilCreate(2, 5, &stencil);

      /* Define the geometry of the stencil */
      {
         int entry;
         int offsets[5][2] = {{0,0}, {-1,0}, {1,0}, {0,-1}, {0,1}};

         for (entry = 0; entry < 5; entry++)
            HYPRE_StructStencilSetElement(stencil, entry, offsets[entry]);
      }
   }


   /* 3. Set up a Struct Matrix */
   { 
      int nentries = 5;  
      int nvalues = nentries*ng1*ng2;

      double *MatValues;      
      int stencil_indices[5];
      
      /* Create an empty matrix object */
      HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &A);

      /* Indicate that the matrix coefficients are ready to be set */
      HYPRE_StructMatrixInitialize(A);
      
      MatValues = (double*) calloc(nvalues, sizeof(double));
      
      for (j = 0; j < nentries; j++) stencil_indices[j] = j;

      /* Set the standard stencil at each grid point */
      for (i = 0; i < nvalues; i += nentries)
      {
         MatValues[i] = 4.0;
         for (int j = 1; j < nentries; j++)
            MatValues[i+j] = -1.0;
      }

      HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, nentries,
                                     stencil_indices, MatValues);
      free(MatValues);
   }
  /* 4. Incorporate the zero boundary conditions: go along each edge of
         the domain and set the stencil entry that reaches to the boundary to
         zero.*/
    
    int nentries = 1;
      
    /*  number of stencil entries times the length
      of one side of my grid box */
    int nvalues_lr = ng2*nentries;   
    int nvalues_ud = ng1*nentries;

    double *values_lr;
    double *values_ud;

    int bc_ilower[2]; 
    int bc_iupper[2];
    
    values_lr = (double*) calloc(nvalues_lr, sizeof(double));
    values_ud = (double*) calloc(nvalues_ud, sizeof(double));
    
    for (j = 0; j < nvalues_lr; j++) values_lr[j] = 0.0;
    for (j = 0; j < nvalues_ud; j++) values_ud[j] = 0.0;
         
    /* X3, North: MPI block on left, Physical boundary on right */
    /* if (pG->rx3_id < 0) { */
    /*    /\* if (pG->rx3_id < 0 && pG->lx3_id >= 0) {  *\/ */
          
    /*   bc_ilower[0] = ig_s; */
    /*   bc_iupper[0] = ig_e; */
    /*   bc_ilower[1] = kg_e; */
    /*   bc_iupper[1] = bc_ilower[1]; */
      
    /*   int stencil_indices[1] = {4}; */

    /*   HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries, */
    /*                                     stencil_indices, values_ud); */
    /* } */
    /* /\*  X3, South: physical boundary on left, MPI block on right  *\/     */
    /* /\* if (pG->rx3_id >= 0 && pG->lx3_id < 0) { *\/ */
    /* if (pG->lx3_id < 0) { */
          
    /*   int bc_ilower[2]; */
    /*   int bc_iupper[2]; */

    /*   bc_ilower[0] = ig_s; */
    /*   bc_iupper[0] = ig_e; */
    /*   bc_ilower[1] = kg_s;   */
    /*   bc_iupper[1] = bc_ilower[1];     */
      

    /*   int stencil_indices[1] = {3}; */
    /*   HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries, */
    /*                                     stencil_indices, values_ud); */
    /* } */

 /* stripe domain in x3 */

    if (pG->rx3_id < 0 && pG->lx3_id < 0) {
          
      int bc_ilower[2];
      int bc_iupper[2];

      bc_ilower[0] = ig_s;

      bc_ilower[1] = kg_s;

      bc_iupper[0] = ig_e;
            
      bc_iupper[1] = kg_s;
      
      int stencil_indices[1] = {3};

      nentries=1;
      HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
                                        stencil_indices, values_ud);

      bc_ilower[0] = ig_s;
      bc_iupper[0] = ig_e;
      bc_ilower[1] = kg_e;
      bc_iupper[1] = bc_ilower[1];
  
      stencil_indices[0] = 4;
      nentries=1;
      HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
                                        stencil_indices, values_ud);
      
    }


    
  /*  X1, West:, physical boundary on left, MPI block on right  */  
    if (pG->rx1_id >= 0 && pG->lx1_id < 0) {
      int bc_ilower[2];
      int bc_iupper[2];

      bc_ilower[0] = ig_s;
      bc_iupper[0] = bc_ilower[0];
      bc_ilower[1] = kg_s;
      bc_iupper[1] = kg_e;
      
      int stencil_indices[1] = {1};
  
      HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
                                        stencil_indices, values_lr);
    }

  /* X1, East: MPI block on left, Physical boundary on right */
    if (pG->rx1_id < 0 && pG->lx1_id >= 0) {
      
      int bc_ilower[2];
      int bc_iupper[2];

      bc_ilower[0] = ig_e;
      bc_iupper[0] = bc_ilower[0];
      bc_ilower[1] = kg_s;
      bc_iupper[1] = kg_e;

      int stencil_indices[1] = {2};

      HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
                                        stencil_indices, values_lr);

    }

   
    

    
    free(values_lr);
    free(values_ud);

     
   {   /* Set up Struct Vectors for b and x */
       
    int   nvalues = ng1*ng2;

    
    
    double *values; 
    values = (double*) calloc(nvalues, sizeof(double));

    /* Create an empty vector object */
    HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &b);
    HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &x);

    /* Indicate that the vector coefficients are ready to be set */
    HYPRE_StructVectorInitialize(b);
    HYPRE_StructVectorInitialize(x);

    for (i = 0; i < nvalues; i ++) values[i] = 1.; //h2;
    
    HYPRE_StructVectorSetBoxValues(b, ilower, iupper, values);

    /* printf("id = %d, %d %d %d\n", my_id, ilower[0],ilower[1],iupper[0],iupper[1] ); */
	   
    
    for (i = 0; i < nvalues; i ++) values[i] = 1;

    HYPRE_StructVectorSetBoxValues(x, ilower, iupper, values);
    /* printf("id = %d, %d %d \n", ilower[0],ilower[1], iupper[0],iupper[1] ); */

     /* this is a collective call finalizing the vector assembly.
         The vector is now ``ready to be used'' */
    HYPRE_StructVectorAssemble(b);
      
    HYPRE_StructVectorAssemble(x);

    free(values);

    int all;
    HYPRE_StructMatrixPrint ("my_matrix.dat",A,all);

    HYPRE_StructVectorPrint ("my_vector_x.dat", x, all);
      
    HYPRE_StructVectorPrint ("my_vector_b.dat", b, all);
    
   }



   
      HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);
      
      HYPRE_StructPCGSetMaxIter(solver, 50 );
      HYPRE_StructPCGSetTol(solver, 1.0e-06 );
      HYPRE_StructPCGSetTwoNorm(solver, 1 );
      HYPRE_StructPCGSetRelChange(solver, 0 );
      HYPRE_StructPCGSetPrintLevel(solver, 2 ); /* print each CG iteration */
      HYPRE_StructPCGSetLogging(solver, 1);

      /* Use symmetric SMG as preconditioner */
      HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
      HYPRE_StructSMGSetMemoryUse(precond, 0);
      HYPRE_StructSMGSetMaxIter(precond, 1);
      HYPRE_StructSMGSetTol(precond, 0.0);
      HYPRE_StructSMGSetZeroGuess(precond);
      HYPRE_StructSMGSetNumPreRelax(precond, 1);
      HYPRE_StructSMGSetNumPostRelax(precond, 1);

      /* Set the preconditioner and solve */
      HYPRE_StructPCGSetPrecond(solver, HYPRE_StructSMGSolve,
                                  HYPRE_StructSMGSetup, precond);

      HYPRE_StructPCGSetup(solver, A, b, x);
      
      HYPRE_StructPCGSolve(solver, A, b, x);

      /* Get some info on the run */
      HYPRE_StructPCGGetNumIterations(solver, &num_iterations);
      HYPRE_StructPCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

      /* Clean up */
      HYPRE_StructPCGDestroy(solver);

printf( "--- testHYPRE_1 done ----" );

}







// #endif


  
//   /* printf("%d %d \n%", iloc - pG->is + pG->Disp[0], *iglob); */   
//   return;
// }


#endif //grav solver
