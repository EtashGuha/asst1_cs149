#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <vector>

#include "CycleTimer.h"

using namespace std;

typedef struct {
  // Control work assignments
  int start, end;

  // Shared by all functions
  double *data;
  double *clusterCentroids;
  int *clusterAssignments;
  double *currCost;
  int M, N, K;
} WorkerArgs;


/**
 * Checks if the algorithm has converged.
 * 
 * @param prevCost Pointer to the K dimensional array containing cluster costs 
 *    from the previous iteration.
 * @param currCost Pointer to the K dimensional array containing cluster costs 
 *    from the current iteration.
 * @param epsilon Predefined hyperparameter which is used to determine when
 *    the algorithm has converged.
 * @param K The number of clusters.
 * 
 * NOTE: DO NOT MODIFY THIS FUNCTION!!!
 */
static bool stoppingConditionMet(double *prevCost, double *currCost,
                                 double epsilon, int K) {
  for (int k = 0; k < K; k++) {
    if (abs(prevCost[k] - currCost[k]) > epsilon)
      return false;
  }
  return true;
}

/**
 * Computes L2 distance between two points of dimensiondist nDim.
 * 
 * @param x Pointer to the beginning of the array representing the first
 *     data point.
 * @param y Poitner to the beginning of the array representing the second
 *     data point.
 * @param nDim The dimensionality (number of elements) in each data point
 *     (must be the same for x and y).
 */
void dist(double *x, double *y, int nDim, double *dist) {
  double accum = 0.0;
  for (int i = 0; i < nDim; i++) {
    accum += pow((x[i] - y[i]), 2);
  }
  *dist = sqrt(accum);
}

double dist_calc(double *x, double *y, int nDim) {
  double accum = 0.0;
  for (int i = 0; i < nDim; i++) {
    accum += pow((x[i] - y[i]), 2);
  }
  return sqrt(accum);
}

/**
 * Assigns each data point to its "closest" cluster centroid.
 */
void computeAssignments(WorkerArgs *const args) {
  double *minDist = new double[args->M];
  
  // Initialize arrays
  for (int m =0; m < args->M; m++) {
    minDist[m] = 1e30;
    args->clusterAssignments[m] = -1;
  }

  for (int k = args->start; k < args->end; k++) {
    int numThreads = 16;
    
    std::vector<std::thread> workers;
    int pointsPerThread = args->M / numThreads;
    int remainingPoints = args->M % numThreads;
    
    for (int t = 0; t < numThreads; t++) {
      int startM = t * pointsPerThread;
      int endM = startM + pointsPerThread;
      if (t == numThreads - 1) {
        endM += remainingPoints;
      }
      
      workers.emplace_back([=]() {
        for (int m = startM; m < endM; m++) {
          double d = dist_calc(&args->data[m * args->N],
                          &args->clusterCentroids[k * args->N], args->N);
          if (d < minDist[m]) {
            minDist[m] = d;
            args->clusterAssignments[m] = k;
          }
        }
      });
    }
    for (auto& worker : workers) {
      worker.join();
    }
  }

  free(minDist);
}

/**
 * Given the cluster assignments, computes the new centroid locations for
 * each cluster.
 */
void computeCentroids(WorkerArgs *const args) {
  int *counts = new int[args->K];

  // Zero things out
  for (int k = 0; k < args->K; k++) {
    counts[k] = 0;
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] = 0.0;
    }
  }


  // Sum up contributions from assigned examples
  for (int m = 0; m < args->M; m++) {
    int k = args->clusterAssignments[m];
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] +=
          args->data[m * args->N + n];
    }
    counts[k]++;
  }

  // Compute means
  for (int k = 0; k < args->K; k++) {
    counts[k] = max(counts[k], 1); // prevent divide by 0
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] /= counts[k];
    }
  }

  free(counts);
}

/**
 * Computes the per-cluster cost. Used to check if the algorithm has converged.
 */
void computeCost(WorkerArgs *const args) {
  double *accum = new double[args->K];

  // Zero things out
  for (int k = 0; k < args->K; k++) {
    accum[k] = 0.0;
  }

  // Sum cost for all data points assigned to centroid
  for (int m = 0; m < args->M; m++) {
    int k = args->clusterAssignments[m];
    accum[k] += dist_calc(&args->data[m * args->N],
                     &args->clusterCentroids[k * args->N], args->N);
  }

  // Update costs
  for (int k = args->start; k < args->end; k++) {
    args->currCost[k] = accum[k];
  }

  free(accum);
}

/**
 * Computes the K-Means algorithm, using std::thread to parallelize the work.
 *
 * @param data Pointer to an array of length M*N representing the M different N 
 *     dimensional data points clustered. The data is layed out in a "data point
 *     major" format, so that data[i*N] is the start of the i'th data point in 
 *     the array. The N values of the i'th datapoint are the N values in the 
 *     range data[i*N] to data[(i+1) * N].
 * @param clusterCentroids Pointer to an array of length K*N representing the K 
 *     different N dimensional cluster centroids. The data is laid out in
 *     the same way as explained above for data.
 * @param clusterAssignments Pointer to an array of length M representing the
 *     cluster assignments of each data point, where clusterAssignments[i] = j
 *     indicates that data point i is closest to cluster centroid j.
 * @param M The number of data points to cluster.
 * @param N The dimensionality of the data points.
 * @param K The number of cluster centroids.
 * @param epsilon The algorithm is said to have converged when
 *     |currCost[i] - prevCost[i]| < epsilon for all i where i = 0, 1, ..., K-1
 */
void kMeansThread(double *data, double *clusterCentroids, int *clusterAssignments,
               int M, int N, int K, double epsilon) {

  double totalTime = CycleTimer::currentSeconds();
  double initTime = CycleTimer::currentSeconds();

  // Used to track convergence
  double *prevCost = new double[K];
  double *currCost = new double[K];

  // The WorkerArgs array is used to pass inputs to and return output from
  // functions.
  WorkerArgs args;
  args.data = data;
  args.clusterCentroids = clusterCentroids;
  args.clusterAssignments = clusterAssignments;
  args.currCost = currCost;
  args.M = M;
  args.N = N;
  args.K = K;

  // Initialize arrays to track cost
  for (int k = 0; k < K; k++) {
    prevCost[k] = 1e30;
    currCost[k] = 0.0;
  }

  initTime = CycleTimer::currentSeconds() - initTime;
  printf("Init time: %.3f ms\n", initTime * 1000);

  double assignmentsTime = 0.0;
  double centroidsTime = 0.0;
  double costTime = 0.0;
  double convergenceTime = 0.0;
  double stoppingConditionTime = 0.0;
  double iterationOverheadTime = 0.0;

  /* Main K-Means Algorithm Loop */
  int iter = 0;
  while (!stoppingConditionMet(prevCost, currCost, epsilon, K)) {
    double iterStart = CycleTimer::currentSeconds();
    
    // Update cost arrays (for checking convergence criteria)
    double convStart = CycleTimer::currentSeconds();
    for (int k = 0; k < K; k++) {
      prevCost[k] = currCost[k];
    }
    convergenceTime += CycleTimer::currentSeconds() - convStart;

    // Setup args struct
    args.start = 0;
    args.end = K;

    double assignStart = CycleTimer::currentSeconds();
    computeAssignments(&args);
    assignmentsTime += CycleTimer::currentSeconds() - assignStart;

    double centroidStart = CycleTimer::currentSeconds();
    computeCentroids(&args);
    centroidsTime += CycleTimer::currentSeconds() - centroidStart;

    double costStart = CycleTimer::currentSeconds();
    computeCost(&args);
    costTime += CycleTimer::currentSeconds() - costStart;

    double stoppingStart = CycleTimer::currentSeconds();
    bool converged = stoppingConditionMet(prevCost, currCost, epsilon, K);
    stoppingConditionTime += CycleTimer::currentSeconds() - stoppingStart;

    iter++;
    
    double iterTime = CycleTimer::currentSeconds() - iterStart;
    double measuredTime = assignmentsTime + centroidsTime + costTime + convergenceTime + stoppingConditionTime;
    iterationOverheadTime += iterTime - (CycleTimer::currentSeconds() - iterStart);
    
    if (iter % 10 == 0) {
      printf("Iter %d: %.3f ms (assign: %.3f, centroid: %.3f, cost: %.3f, conv: %.3f, stop: %.3f)\n", 
             iter, iterTime * 1000, 
             (CycleTimer::currentSeconds() - assignStart) * 1000,
             (CycleTimer::currentSeconds() - centroidStart) * 1000,
             (CycleTimer::currentSeconds() - costStart) * 1000,
             convergenceTime * 1000,
             stoppingConditionTime * 1000);
    }
  }

  totalTime = CycleTimer::currentSeconds() - totalTime;
  
  printf("\nTiming breakdown:\n");
  printf("Total time: %.3f ms\n", totalTime * 1000);
  printf("Init time: %.3f ms (%.1f%%)\n", initTime * 1000, (initTime/totalTime)*100);
  printf("Assignments: %.3f ms (%.1f%%)\n", assignmentsTime * 1000, (assignmentsTime/totalTime)*100);
  printf("Centroids: %.3f ms (%.1f%%)\n", centroidsTime * 1000, (centroidsTime/totalTime)*100);
  printf("Cost: %.3f ms (%.1f%%)\n", costTime * 1000, (costTime/totalTime)*100);
  printf("Convergence: %.3f ms (%.1f%%)\n", convergenceTime * 1000, (convergenceTime/totalTime)*100);
  printf("Stopping condition: %.3f ms (%.1f%%)\n", stoppingConditionTime * 1000, (stoppingConditionTime/totalTime)*100);
  printf("Iterations: %d\n", iter);
  
  double accountedTime = initTime + assignmentsTime + centroidsTime + costTime + convergenceTime + stoppingConditionTime;
  double unaccountedTime = totalTime - accountedTime;
  printf("\nAccounted time: %.3f ms (%.1f%%)\n", accountedTime * 1000, (accountedTime/totalTime)*100);
  printf("Unaccounted time: %.3f ms (%.1f%%)\n", unaccountedTime * 1000, (unaccountedTime/totalTime)*100);

  free(currCost);
  free(prevCost);
}
