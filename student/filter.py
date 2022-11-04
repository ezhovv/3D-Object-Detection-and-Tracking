# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        self.dim_state = params.dim_state
        self.dt = params.dt
        self.q = params.q

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        dt = self.dt
        F_matrix = np.matrix(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ]
        )
        return F_matrix
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        dt = self.dt
        q = self.q

        # process noise covariance coefficients
        q1 = ((dt**3)/3) * q
        q2 = ((dt**2)/2) * q
        q3 = dt*q
        Q_matrix = np.matrix([
                [q1, 0, 0, q2, 0, 0],
                [0, q1, 0, 0, q2, 0],
                [0, 0, q1, 0, 0, q2],
                [q2, 0, 0, q3, 0, 0],
                [0, q2, 0, 0, q3, 0],
                [0, 0, q2, 0, 0, q3]
        ])
        return Q_matrix
        
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############

        # get current state vector
        x = track.x

        # get estimation error covariance matrix P
        P = track.P

        # calculate system matrix F
        F = self.F()

        # calculate process noise covariance matrix Q
        Q = self.Q()

        # predict state vector for next timestep
        x = F*x

        # predict estimation error covariance matrix for next timestep
        P = F*P*F.transpose() + Q

        # save predicted state vector
        track.set_x(x)

        # save predicted estimation error covariance matrix
        track.set_P(P)


        
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############

        # get state vector
        x = track.x

        # get estimation error covariance matrix
        P = track.P

        # get Jacobian H 
        H = meas.sensor.get_H(x)

        # calculate residual gamma between sensor measurement and predicted measurement
        gamma = self.gamma(track, meas) 

        # caclulate residual covariance matrix S
        S = self.S(track,meas,H)

        # calculate Kalman filter gain matrix
        K = P*H.transpose()*np.linalg.inv(S)

        # measurement update 
        x = x + K*gamma

        # get identity matrix
        I = np.identity(self.dim_state)

        # update noise covariance matrix
        P = (I - K*H) *P

        # save measurement update of state vector
        track.set_x(x)

        # save measurement update of the estimation error covariance matrix
        track.set_P(P)





        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############
        # get sensor measuremment
        z = meas.z
        # get state vector for current timestep
        x = track.x

        # calculate nonlinear measurement expectation value h(x) 
        hx = meas.sensor.get_hx(x)

        # calculate residual gamma between sensor measurement and predicted measurement
        gamma = z- hx
        return gamma
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############
        # get estimation error covariance matrix P
        P = track.P

        # get measurement nosie covariance matrix R 
        R = meas.R

        # calculate residual error covariance matrix S
        S = H*P*H.transpose() + R
        return S
        
        ############
        # END student code
        ############ 