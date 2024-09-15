# Autonomous Robotic Navigation System

This repository contains the code and implementation details for the Autonomous Robotic Navigation System developed using the Robile platform and ROS. The project includes global path planning with the A* algorithm, local obstacle avoidance using Potential Fields, and localization through a Particle Filter. Additionally, the robot autonomously explores unknown environments via frontier-based exploration. 

## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Team Members](#team-members)
- [References](#references)
- [Appendix](#appendix)

## Introduction

Autonomous mobile robots are crucial in fields such as logistics, exploration, and surveillance, where navigation in unknown environments is essential. This project implements an autonomous navigation system on the Robile platform using the Robot Operating System (ROS). The system incorporates:
- A* Algorithm for global pathfinding.
- Potential Fields for local dynamic obstacle avoidance.
- Particle Filter-based localization for accurate positioning.
- Frontier-based exploration for autonomous environment exploration.

## Key Features

- **Global Path Planning (A* Algorithm)**: Computes the optimal path from a start to a goal position in a 2D occupancy grid while avoiding obstacles.
- **Local Obstacle Avoidance (Potential Fields)**: Balances attractive forces towards the goal and repulsive forces away from obstacles, ensuring smooth navigation.
- **Particle Filter Localization**: Accurately tracks the robot's position using sensor data and odometry updates.
- **Frontier-Based Exploration**: Enables the robot to autonomously explore unknown areas by detecting frontiers between explored and unexplored spaces.


## Project Structure

- **Path and Motion Planning**
  - *Global Planner (A*)*: Computes the shortest path to a goal.
  - *Local Planner (Potential Fields)*: Dynamically avoids obstacles.
  
- **Localization**
  - *Particle Filter*: Estimates the robot's pose using sensor data and odometry.
  
- **Exploration**
  - *Frontier-Based Exploration*: Directs the robot to explore new areas autonomously.

- **Code Structure**:
    - `src/`: Source code files for path planning, localization, and exploration.
    - `modules/`: Modules created for the robot and simulation.
    - `launch/`: ROS launch files to initialize the system.
    - `worlds/`: world files created for the robot and simulation.
    - `maps/`: map files created for the robot and simulation.


## Team Members

- **Amol Tatkari**: Particle Filter implementation, Debugging, Testing, Technical Report
- **Ayushi Arora**: Exploration implementation, Debugging, Testing, Technical Report
- **Behrouz Ghamkar**: A* and Potential Fields Planner, Debugging, Testing, Repository Management, Technical Report
- **Ujjwal Patil**: A* and Potential Fields Planner, Robot Communication, Debugging, Testing, Technical Report

## References
- *Lecture slides of Autonomous Mobile Robots, Dr. Alex Mitrevski
- *Introduction to Autonomous Mobile Robots, 2nd edition.
- *Probabilistic Robotics.
- *Springer Handbook of Robotics, 2nd edition.

## Appendix


### Videos

Demonstration videos of the project:

- [Video 1](https://)
- [Video 2](https://)
- [Video 3](https://)
