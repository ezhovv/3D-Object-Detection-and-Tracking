# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params 

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        
    def associate(self, track_list, meas_list, KF):
             
        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############
        
        # reset matrix
        self.association_matrix = np.matrix([]) 
        # reset lists
        self.unassigned_tracks = [] 
        self.unassigned_meas = []

        
        # create lists of unassigned tracks and measurements
        if len(track_list) > 0:
            self.unassigned_tracks = list(range(len(track_list)))
        if len(meas_list) > 0:
            self.unassigned_meas = list(range(len(meas_list)))
        
        if len(meas_list) > 0 and len(track_list) > 0: 
            # initialize n x m association matrix with inf values
            self.association_matrix = np.inf * np.ones((len(track_list),len(meas_list)))
        
        # loop over all tracks and all measurements 
        # to assign values to entries of a matrix
        for i in range(len(track_list)):
            track = track_list[i]
            for j in range(len(meas_list)):
                meas = meas_list[j]
                dist = self.MHD(track, meas, KF)
                # implement gating to remove unlikely measurements
                if self.gating(dist, meas.sensor):
                    # update association matrix
                    self.association_matrix[i,j] = dist
        ############
        # END student code
        ############ 
                
    def get_closest_track_and_meas(self):
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        # define association matrix
        A = self.association_matrix
        # if initial values weren't changed return nan
        if np.min(A) == np.inf:
            return np.nan, np.nan

        # retrieve indices ofsmallest entry in association matrix
        i_min, j_min = np.where(A == np.min(A))

        # update matrix by deleting row and column with smallest distance
        A = np.delete(A, i_min[0], 0)
        A = np.delete(A, j_min[0], 1)
        self.association_matrix = A

        # update track corresponding to smallest distance
        update_track = self.unassigned_tracks[i_min[0]]
        update_meas = self.unassigned_meas[j_min[0]]
        
        # remove selected track and measurement from lists
        self.unassigned_tracks.remove(update_track) 
        self.unassigned_meas.remove(update_meas)
            
        ############
        # END student code
        ############ 
        return update_track, update_meas     

    def gating(self, MHD, sensor): 
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############
        
        # find gate limit
        limit = chi2.ppf(params.gating_threshold, df = sensor.dim_meas)

        # check if Mahalanobis distance lies inside the gate
        within_limit = bool(MHD < limit)

        return within_limit    
        
        ############
        # END student code
        ############ 
        
    def MHD(self, track, meas, KF):
        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############
        
        # get Jacobian matrix  of measurement function
        H = meas.sensor.get_H(track.x)

        # calculate residual gamma between sensor measuremment and prediction
        gamma = KF.gamma(track, meas)

        # calculate residual error covariance matrix S
        S = KF.S(track, meas, H)

        # calculate Mahalanobis distance btwn track and measurement
        MHD = gamma.transpose()*np.linalg.inv(S)*gamma

        return MHD


        
        ############
        # END student code
        ############ 
    
    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
    
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)