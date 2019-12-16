/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 1000;  // TODO: Set the number of particles
  // initialize particles
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  for (int i = 0; i < num_particles; i++)
  {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1;

    particles.push_back(p);
    weights.push_back(1.0);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);
  
  for (int i = 0; i < particles.size(); i++)
  {
    Particle& curr_p = particles[i];
    if (yaw_rate == 0)
    {
      curr_p.x += velocity * delta_t * cos(curr_p.theta);
      curr_p.y += velocity * delta_t * sin(curr_p.theta);
    }
    else
    {
      double theta_f = curr_p.theta + yaw_rate * delta_t;
      curr_p.x += velocity * (sin(theta_f) - sin(curr_p.theta)) / yaw_rate;
      curr_p.y += velocity * (cos(curr_p.theta) - cos(theta_f)) / yaw_rate;
      curr_p.theta = theta_f;
    }
    
    // add Gausian noise
    curr_p.x += dist_x(gen);
    curr_p.y += dist_y(gen);
    curr_p.theta += dist_theta(gen);
  }
}

int ParticleFilter::getAssociation(const double &x_m, 
                                   const double &y_m,
                                   const Map &map_landmarks)
{
  int association = -1;
  double base_dist = std::numeric_limits<double>::max();
  for (int i = 0; i < map_landmarks.landmark_list.size(); i++)
  {
    double x_lm = map_landmarks.landmark_list[i].x_f;
    double y_lm = map_landmarks.landmark_list[i].y_f;

    double curr_dist = dist(x_m, y_m, x_lm, y_lm);
    if (curr_dist < base_dist)
    {
      association = map_landmarks.landmark_list[i].id_i;
      base_dist = curr_dist;
    }
  }  
  
  //debug
//   if (association == -1)
//   {
//     printf("map landmarks size: %d, x_m: %f, y_m: %f \n", map_landmarks.landmark_list.size(), x_m, y_m);
//   }
  

  return association;
}

void ParticleFilter::dataAssociation(const Map &map_landmarks, 
                                     const vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (int i = 0; i < particles.size(); i++)
  {
    Particle& curr_p = particles[i];
    double x_p = curr_p.x;
    double y_p = curr_p.y;
    double theta_p = curr_p.theta;

    std::vector<int> associations;
    std::vector<double> sense_xs;
    std::vector<double> sense_ys;
    for (int j = 0; j < observations.size(); j++)
    {
      double x_obs = observations[j].x;
      double y_obs = observations[j].y;

      double x_m = x_p + (cos(theta_p) * x_obs) - (sin(theta_p) * y_obs);
      double y_m = y_p + (sin(theta_p) * x_obs) + (cos(theta_p) * y_obs);
      int association = getAssociation(x_m, y_m, map_landmarks);
      //debug
      if (association == -1)
      {
        printf("theta_p: %f, cos: %f, sin: %f, x_obs: %f, y_obs: %f, x_p: %f, y_p: %f  \n", theta_p, cos(theta_p), sin(theta_p), x_obs, y_obs, x_p, y_p);
      }

      associations.push_back(association);
      sense_xs.push_back(x_m);
      sense_ys.push_back(y_m);
    }

    SetAssociations(curr_p, associations, sense_xs, sense_ys);
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  if (observations.size() == 0)
    std::cout << "Empty observations.." << std::endl;
  if (map_landmarks.landmark_list.size() == 0)
    std::cout << "Empty map landmarks.." << std::endl;

  // initialize associations map
  for (int i = 0; i < map_landmarks.landmark_list.size(); i++)
  {
    double x_lm = map_landmarks.landmark_list[i].x_f;
    double y_lm = map_landmarks.landmark_list[i].y_f;
    int id = map_landmarks.landmark_list[i].id_i;

    LandmarkObs lm;
    lm.x = x_lm;
    lm.y = y_lm;
    lm.id = id;

    associations_map[id] = lm;
    //printf("landmark id: %d, x: %f, y: %f \n", id, lm.x, lm.y);
  }
  
  dataAssociation(map_landmarks, observations);

  for (int i = 0; i < particles.size(); i++)
  {
    Particle& curr_p = particles[i];
    curr_p.weight = 1;
    for (int j = 0; j < curr_p.associations.size(); j++)
    {
      double x_obs = curr_p.sense_x[j];
      double y_obs = curr_p.sense_y[j];

      int lm_id = curr_p.associations[j];
      if (associations_map.find(lm_id) == associations_map.end())
      {
        std::cout << "INVALID association id : " << lm_id << std::endl;
        continue;
      }
      LandmarkObs lm = associations_map[lm_id];
      curr_p.weight *= multiv_prob(std_landmark[0], std_landmark[1], x_obs, y_obs, lm.x, lm.y);
    }
  }


}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  weights.clear();
  for (int i = 0; i < particles.size(); i++)
  {
    weights.push_back(particles[i].weight);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d(std::begin(weights), std::end(weights));

  std::vector<Particle> resample_particles;
  for (int i = 0; i < num_particles; i++)
  {
    int index = d(gen);
    resample_particles.push_back(particles[index]);
  }

  particles = resample_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}