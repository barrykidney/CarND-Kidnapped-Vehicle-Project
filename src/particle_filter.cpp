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

// #include "helper_functions.h"

using namespace std; 

// using std::string;
// using std::vector;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  num_particles = 10;  // TODO: Set the number of particles

  double std_x = std[0]; 
  double std_y = std[1];
  double std_theta = std[2];

  std::normal_distribution<double> init_dist_x(x, std_x);
  std::normal_distribution<double> init_dist_y(y, std_y);
  std::normal_distribution<double> init_dist_theta(theta, std_theta);

  for (int i = 0; i < num_particles; ++i) {
    Particle new_particle;

    new_particle.id = i;
    new_particle.x = init_dist_x(gen);
    new_particle.y = init_dist_y(gen);
    new_particle.theta = init_dist_theta(gen);
    new_particle.weight = 1.0; 

    particles.push_back(new_particle);
    weights.push_back(1.0);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {

  for (unsigned int i = 0; i < particles.size(); i++) {

    if(abs(yaw_rate) > 0.00001) {

      double predicted_x = (velocity/yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t))- sin(particles[i].theta));
      double predicted_y = (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
      double predicted_heading = yaw_rate * delta_t;

      std::normal_distribution<double> predict_dist_x(predicted_x, std_pos[0]);
      std::normal_distribution<double> predict_dist_y(predicted_y, std_pos[1]);
      std::normal_distribution<double> predict_dist_theta(predicted_heading, std_pos[2]);

      particles[i].x += predict_dist_x(gen);
      particles[i].y += predict_dist_y(gen);
      particles[i].theta += predict_dist_theta(gen);

    } else {

      double predicted_x = velocity * delta_t * cos(particles[i].theta);
      double predicted_y = velocity * delta_t * sin(particles[i].theta);

      std::normal_distribution<double> predict_dist_x(predicted_x, std_pos[0]);
      std::normal_distribution<double> predict_dist_y(predicted_y, std_pos[1]);
      std::normal_distribution<double> predict_dist_theta(0, std_pos[2]);

      particles[i].x += predict_dist_x(gen);
      particles[i].y += predict_dist_y(gen);
      particles[i].theta += predict_dist_theta(gen);
    }
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {

  for (unsigned int observ_idx = 0; observ_idx < observations.size(); observ_idx++) {
    double obs_x = observations[observ_idx].x;
    double obs_y = observations[observ_idx].y;
    double minimum_distance = INFINITY;
    int id_of_minimum_predicted_landmark = -1;

    for (unsigned int prediction_idx = 0; prediction_idx < predicted.size(); prediction_idx++) {
      double predict_x = predicted[prediction_idx].x;
      double predict_y = predicted[prediction_idx].y;
      double current_distance = dist(obs_x, obs_y, predict_x, predict_y);
      if (current_distance < minimum_distance) {
        minimum_distance = current_distance;
        id_of_minimum_predicted_landmark = prediction_idx;
      }
    }
    observations[observ_idx].id = id_of_minimum_predicted_landmark;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

  // move as many of the calculations as possible outside the loop to improve performance
  double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
  double sig_x = 2 * pow(std_landmark[0], 2);
  double sig_y = 2* pow(std_landmark[1], 2);

  for (int particle_idx = 0; particle_idx < num_particles; particle_idx++) {
    Particle current_particle = particles[particle_idx];

    vector<LandmarkObs> landmark_vector;
    for (unsigned int landmark_idx = 0; landmark_idx < map_landmarks.landmark_list.size(); ++landmark_idx) {
      LandmarkObs currnet_landmark;
      currnet_landmark.x = map_landmarks.landmark_list[landmark_idx].x_f;
      currnet_landmark.y = map_landmarks.landmark_list[landmark_idx].y_f;
      currnet_landmark.id = map_landmarks.landmark_list[landmark_idx].id_i;

      if (dist(current_particle.x, current_particle.y, currnet_landmark.x, currnet_landmark.y) <= sensor_range) {
        landmark_vector.push_back(currnet_landmark);
      }
    }

    vector<LandmarkObs> transformed_observations;
    for (unsigned int observ_idx = 0; observ_idx < observations.size(); observ_idx++) {
      LandmarkObs current_observ = observations[observ_idx];

      LandmarkObs transformed_observation;
      transformed_observation.id = current_observ.id;
      transformed_observation.x = current_observ.x*cos(current_particle.theta) - current_observ.y*sin(current_particle.theta) + current_particle.x;
      transformed_observation.y = current_observ.x*sin(current_particle.theta) + current_observ.y*cos(current_particle.theta) + current_particle.y;
      transformed_observations.push_back(transformed_observation);

      dataAssociation(landmark_vector, transformed_observations);

      double particle_probability = 1.0;      
      for (unsigned int observ_idx = 0; observ_idx < transformed_observations.size(); observ_idx++) {

        LandmarkObs transformed_observation = transformed_observations[observ_idx];
        LandmarkObs associated_landmark = landmark_vector[transformed_observation.id];
        particle_probability *= gauss_norm * exp(-(pow(transformed_observation.x - associated_landmark.x, 2)/sig_x + pow(transformed_observation.y - associated_landmark.y, 2) / sig_y ));
      }
      particles[particle_idx].weight = particle_probability;
      weights[particle_idx] = particle_probability;
    }
  }
}

void ParticleFilter::resample() {

  vector<Particle> resampled_particles;
  int idx = rand() % num_particles;

  double max_weight = 0;
   for (int particle_idx = 0; particle_idx < num_particles; particle_idx++) {
    if (weights[particle_idx] > max_weight) {
      max_weight = weights[particle_idx];
    }
  } 

  double beta = 0.0;
  for (int particle_idx = 0; particle_idx < num_particles; particle_idx++) {
    bool picked = false;
    double random_number = (double(rand()) / double(RAND_MAX));

    beta += random_number * max_weight * 2.0;
    while (picked == false) {
      if (weights[idx] < beta) {
        beta -= weights[idx];
        idx = (idx+1) % num_particles;
      } else {
        resampled_particles.push_back(particles[idx % num_particles]);
        picked = true;
      }
    }
  }
  particles = resampled_particles;  
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