# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.

import math

from collections import defaultdict
from typing import List, Dict, Tuple
import sys
import numpy as np
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import MetricData, EvalBox
from nuscenes.eval.common.utils import center_distance
from nuscenes.eval.detection.constants import DETECTION_NAMES, ATTRIBUTE_NAMES, TP_METRICS

class DetectionConfig:
    """ Data class that specifies the detection evaluation settings. """

    def __init__(self,
                 class_range: Dict[str, int],
                 dist_fcn: str,
                 dist_ths: List[float],
                 dist_th_tp: float,
                 min_recall: float,
                 min_precision: float,
                 max_boxes_per_sample: int,
                 mean_ap_weight: int):

        assert set(class_range.keys()) == set(DETECTION_NAMES), "Class count mismatch."
        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight

        self.class_names = self.class_range.keys()

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'class_range': self.class_range,
            'dist_fcn': self.dist_fcn,
            'dist_ths': self.dist_ths,
            'dist_th_tp': self.dist_th_tp,
            'min_recall': self.min_recall,
            'min_precision': self.min_precision,
            'max_boxes_per_sample': self.max_boxes_per_sample,
            'mean_ap_weight': self.mean_ap_weight
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized dictionary. """
        return cls(content['class_range'],
                   content['dist_fcn'],
                   content['dist_ths'],
                   content['dist_th_tp'],
                   content['min_recall'],
                   content['min_precision'],
                   content['max_boxes_per_sample'],
                   content['mean_ap_weight'])

    @property
    def dist_fcn_callable(self):
        """ Return the distance function corresponding to the dist_fcn string. """
        if self.dist_fcn == 'center_distance':
            return center_distance
        else:
            raise Exception('Error: Unknown distance function %s!' % self.dist_fcn)


class DetectionMetricData(MetricData):
    """ This class holds accumulated and interpolated data required to calculate the detection metrics. """

    nelem = 101

    def __init__(self,
                 recall: np.array,
                 recall_crit: np.array,
                 precision: np.array,
                 precision_crit: np.array,
                 confidence: np.array,
                 trans_err: np.array,
                 vel_err: np.array,
                 scale_err: np.array,
                 orient_err: np.array,
                 attr_err: np.array):

        # Assert lengths.
        assert len(recall) == self.nelem
        assert len(recall_crit) == self.nelem
        assert len(precision) == self.nelem
        assert len(precision_crit) == self.nelem
        assert len(confidence) == self.nelem
        assert len(trans_err) == self.nelem
        assert len(vel_err) == self.nelem
        assert len(scale_err) == self.nelem
        assert len(orient_err) == self.nelem
        assert len(attr_err) == self.nelem

        # Assert ordering.
        assert all(confidence == sorted(confidence, reverse=True))  # Confidences should be descending.
        assert all(recall == sorted(recall))  # Recalls should be ascending.
        assert all(recall_crit == sorted(recall_crit))  # Recalls should be ascending.

        # Set attributes explicitly to help IDEs figure out what is going on.
        self.recall = recall
        self.recall_crit = recall_crit
        self.precision = precision
        self.precision_crit = precision_crit
        self.confidence = confidence
        self.trans_err = trans_err
        self.vel_err = vel_err
        self.scale_err = scale_err
        self.orient_err = orient_err
        self.attr_err = attr_err

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    @property
    def max_recall_ind(self):
        """ Returns index of max recall achieved. """

        # Last instance of confidence > 0 is index of max achieved recall.
        non_zero = np.nonzero(self.confidence)[0]
        if len(non_zero) == 0:  # If there are no matches, all the confidence values will be zero.
            max_recall_ind = 0
        else:
            max_recall_ind = non_zero[-1]

        return max_recall_ind

    @property
    def max_recall(self):
        """ Returns max recall achieved. """

        return self.recall[self.max_recall_ind]

    def serialize(self):
        """ Serialize instance into json-friendly format. """
        return {
            'recall': self.recall.tolist(),
            'precision': self.precision.tolist(),
            'confidence': self.confidence.tolist(),
            'trans_err': self.trans_err.tolist(),
            'vel_err': self.vel_err.tolist(),
            'scale_err': self.scale_err.tolist(),
            'orient_err': self.orient_err.tolist(),
            'attr_err': self.attr_err.tolist(),
        }

    @classmethod
    def deserialize(cls, content: dict, nusc):
        """ Initialize from serialized content. """
        return cls(recall=np.array(content['recall']),
                   precision=np.array(content['precision']),
                   confidence=np.array(content['confidence']),
                   trans_err=np.array(content['trans_err']),
                   vel_err=np.array(content['vel_err']),
                   scale_err=np.array(content['scale_err']),
                   orient_err=np.array(content['orient_err']),
                   attr_err=np.array(content['attr_err']))

    @classmethod
    def no_predictions(cls):
        """ Returns a md instance corresponding to having no predictions. """
        return cls(recall=np.linspace(0, 1, cls.nelem),
                   precision=np.zeros(cls.nelem),
                   recall_crit=np.zeros(cls.nelem), #modified, to be rechecked if it gives mistakes
                   precision_crit=np.zeros(cls.nelem),#modified, to be rechecked if it gives mistakes
                   confidence=np.zeros(cls.nelem),
                   trans_err=np.ones(cls.nelem),
                   vel_err=np.ones(cls.nelem),
                   scale_err=np.ones(cls.nelem),
                   orient_err=np.ones(cls.nelem),
                   attr_err=np.ones(cls.nelem))

    @classmethod
    def random_md(cls):
        """ Returns an md instance corresponding to a random results. """
        return cls(recall=np.linspace(0, 1, cls.nelem),
                   precision=np.random.random(cls.nelem),
                   confidence=np.linspace(0, 1, cls.nelem)[::-1],
                   trans_err=np.random.random(cls.nelem),
                   vel_err=np.random.random(cls.nelem),
                   scale_err=np.random.random(cls.nelem),
                   orient_err=np.random.random(cls.nelem),
                   attr_err=np.random.random(cls.nelem))


class DetectionMetrics:
    """ Stores average precision and true positive metric results. Provides properties to summarize. """

    def __init__(self, cfg: DetectionConfig):

        self.cfg = cfg
        self._label_aps = defaultdict(lambda: defaultdict(float))
        self._label_tp_errors = defaultdict(lambda: defaultdict(float))
        self.eval_time = None
        self._label_aps_crit= defaultdict(lambda: defaultdict(float))

    def add_label_ap(self, detection_name: str, dist_th: float, ap: float) -> None:
        self._label_aps[detection_name][dist_th] = ap

    def add_label_ap_crit(self, detection_name: str, dist_th: float, ap_crit: float) -> None:
        self._label_aps_crit[detection_name][dist_th] = ap_crit
        
    def get_label_ap(self, detection_name: str, dist_th: float) -> float:
        return self._label_aps[detection_name][dist_th]

    def get_label_ap_crit(self, detection_name: str, dist_th: float) -> float:
        return self._label_aps_crit[detection_name][dist_th]
    
    def add_label_tp(self, detection_name: str, metric_name: str, tp: float):
        self._label_tp_errors[detection_name][metric_name] = tp

    def get_label_tp(self, detection_name: str, metric_name: str) -> float:
        return self._label_tp_errors[detection_name][metric_name]

    def add_runtime(self, eval_time: float) -> None:
        self.eval_time = eval_time

    @property
    def mean_dist_aps(self) -> Dict[str, float]:
        """ Calculates the mean over distance thresholds for each label. """
        return {class_name: np.mean(list(d.values())) for class_name, d in self._label_aps.items()}

    @property
    def mean_ap(self) -> float:
        """ Calculates the mean AP by averaging over distance thresholds and classes. """
        return float(np.mean(list(self.mean_dist_aps.values())))

    @property
    def tp_errors(self) -> Dict[str, float]:
        """ Calculates the mean true positive error across all classes for each metric. """
        errors = {}
        for metric_name in TP_METRICS:
            class_errors = []
            for detection_name in self.cfg.class_names:
                class_errors.append(self.get_label_tp(detection_name, metric_name))

            errors[metric_name] = float(np.nanmean(class_errors))

        return errors

    @property
    def tp_scores(self) -> Dict[str, float]:
        scores = {}
        tp_errors = self.tp_errors
        for metric_name in TP_METRICS:

            # We convert the true positive errors to "scores" by 1-error.
            score = 1.0 - tp_errors[metric_name]

            # Some of the true positive errors are unbounded, so we bound the scores to min 0.
            score = max(0.0, score)

            scores[metric_name] = score

        return scores

    @property
    def nd_score(self) -> float:
        """
        Compute the nuScenes detection score (NDS, weighted sum of the individual scores).
        :return: The NDS.
        """
        # Summarize.
        total = float(self.cfg.mean_ap_weight * self.mean_ap + np.sum(list(self.tp_scores.values())))

        # Normalize.
        total = total / float(self.cfg.mean_ap_weight + len(self.tp_scores.keys()))

        return total

    def serialize(self):
        return {
            'label_aps': self._label_aps,
            'label_aps_crit': self._label_aps,
            'mean_dist_aps': self.mean_dist_aps,
            'mean_ap': self.mean_ap,
            'label_tp_errors': self._label_tp_errors,
            'tp_errors': self.tp_errors,
            'tp_scores': self.tp_scores,
            'nd_score': self.nd_score,
            'eval_time': self.eval_time,
            'cfg': self.cfg.serialize()
        }

    @classmethod
    def deserialize(cls, content: dict, nusc):
        """ Initialize from serialized dictionary. """

        cfg = DetectionConfig.deserialize(content['cfg'])

        metrics = cls(cfg=cfg)
        metrics.add_runtime(content['eval_time'])

        for detection_name, label_aps in content['label_aps'].items():
            for dist_th, ap in label_aps.items():
                metrics.add_label_ap(detection_name=detection_name, dist_th=float(dist_th), ap=float(ap))

        for detection_name, label_aps_crit in content['label_aps_crit'].items():
            for dist_th, ap in label_aps_crit.items():
                metrics.add_label_ap(detection_name=detection_name, dist_th=float(dist_th), ap=float(ap))

        for detection_name, label_tps in content['label_tp_errors'].items():
            for metric_name, tp in label_tps.items():
                metrics.add_label_tp(detection_name=detection_name, metric_name=metric_name, tp=float(tp))

        return metrics

    def __eq__(self, other):
        eq = True
        eq = eq and self._label_aps == other._label_aps
        eq = eq and self._label_tp_errors == other._label_tp_errors
        eq = eq and self.eval_time == other.eval_time
        eq = eq and self.cfg == other.cfg

        return eq


class DetectionBox(EvalBox):
    MAX_DISTANCE_OBJ : float =9999.0
    MAX_DISTANCE_INTERSECT : float =9999.0
    MAX_TIME_INTERSECT_OBJ : float =9999.0

    """ Data class used during detection evaluation. Can be a prediction or ground truth."""
    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 ego_translation: [float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters. NOT USED????
                 num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
                 detection_name: str = 'car',  # The class name used in the detection challenge.
                 detection_score: float = -1.0,  # GT samples do not have a score.
                 attribute_name: str = '',  # Box attribute. Each box can have at most 1 attribute.
                 ego_speed: float = 9999.0,  # default speed di ego; should be always overwritten
                 nusc = None
                ):

        super().__init__(sample_token, translation, size, rotation, velocity, ego_translation, num_pts)

        assert detection_name is not None, 'Error: detection_name cannot be empty!'
        assert detection_name in DETECTION_NAMES, 'Error: Unknown detection_name %s' % detection_name

        assert attribute_name in ATTRIBUTE_NAMES or attribute_name == '', \
            'Error: Unknown attribute_name %s' % attribute_name

        assert type(detection_score) == float, 'Error: detection_score must be a float!'
        assert not np.any(np.isnan(detection_score)), 'Error: detection_score may not be NaN!'
        # Assign.
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name
        #print("Trying to add ego data")
        #NOW I ADD EGO DATA
        #get ego values (speed) 
        current_sample=nusc.get('sample',sample_token)
        ego_LIDAR=nusc.get("sample_data", current_sample["data"]["LIDAR_TOP"]) #prendo il prev sample        
        ego_pose=nusc.get('ego_pose', ego_LIDAR['ego_pose_token'])
        ego_coordinates=ego_pose['translation']
        ego_timestamp=current_sample['timestamp']
        #print(ego_timestamp)
        #get prev or next, for speed computation
        #prev_sample_from_nuscene=nuscenes.get('sample',s1['prev'])
        if(current_sample['prev']==''): # if first sample, speed is calculated using the second frame
            next_sample=nusc.get('sample',current_sample['next'])
            #calcolo la speed in modo complicato con s_next
            next_LIDAR=nusc.get("sample_data", next_sample["data"]["LIDAR_TOP"]) #prendo il prev sample        
            next_pose=nusc.get('ego_pose', next_LIDAR['ego_pose_token'])
            next_coordinates=next_pose['translation']
            next_timestamp=next_sample['timestamp']
            speed=self.speed(next_coordinates, next_timestamp, ego_coordinates, ego_timestamp)
        else: #calcolo la speed in modo normale
            prev_sample=nusc.get('sample',current_sample['prev']) #speed is calculated wrt the previous frame
            prev_LIDAR=nusc.get("sample_data", prev_sample["data"]["LIDAR_TOP"]) #prendo il prev sample        
            prev_pose=nusc.get('ego_pose', prev_LIDAR['ego_pose_token'])
            prev_coordinates=prev_pose['translation']
            prev_timestamp=prev_sample['timestamp']
            speed=self.speed(ego_coordinates, ego_timestamp, prev_coordinates, prev_timestamp)
       
        self.ego_speed=speed
        #=======================================================
        #COMPUTE CRIT_R, CRIT_T, CRIT
        #FORMULAS SHOULD BE AS IN THE PAPER (TO BE REVISED)
        #=======================================================

        # Set defaults to maximum criticality. If we don't have some data, e.g., speed or position
        # then consider maximum criticality for safety
        crit_d = 1.0
        crit_r = 1.0
        crit_t = 1.0

        # Get position data of ego and object B
        B_Y=self.translation[1]
        B_X=self.translation[0]
        ego_x=ego_coordinates[0]
        ego_y=ego_coordinates[1]
        self.ego_translation=ego_coordinates #ego_translation è inutile di fatto
#       print("-------" + sample_token)
#       print(B_X)
#       print(B_Y)
#       print(ego_x)
#       print(ego_y)

        # 1) Calculate criticality with respect to static distance from the object (crit_d)
        #    This values does not use the speed, therefore it should be always possible to calculate it.
        d_ego_B = math.sqrt((ego_x-B_X)**2 + (ego_y-B_Y)**2)
        # criticality wrt to distance of ego from object
        # calculated as a 2nd grade equation (parabola) passing from (0,1) and (MAX_DISTANCE_OBJ,0)
        # the value is 1.0 when r=0 and decreases until 0.0 when r=MAX_DISTANCE_OBJ
        crit_d = -((d_ego_B ** 2) / (self.MAX_DISTANCE_OBJ ** 2)) + 1
#       print(crit_d);
        crit_d = (crit_d, 0)[crit_d < 0]  # set to zero if negative
#       print('crit_d '+str(crit_d));

        # 2) Calculate criticality with respect to distance from intersection point (crit_r)
        # 3) Calculate criticality with respect to time for B to reach the intersection point (crit_t)

        # Calculate relative (apparent) velocity of object
        v_By_egoy=  float(self.velocity[1])-float(self.ego_speed[1])
        v_Bx_egox=  float(self.velocity[0])-float(self.ego_speed[0])

        if( math.isnan(v_Bx_egox) or math.isnan(v_By_egoy) ): #CAPITA MA NON E' COSA CHE POSSIAMO RISOLVERE NOI
            # For some reason we can't calculate the speed. In this case do nothing,
            # crit_r e crit_t use the default value (1.0, maximum criticality)
#           print("detection/data_classes.py line 440, speed is NaN, use max crit_d/crit_t -- check if OK")
            mettoQualcosaqua=0.0
        elif( v_Bx_egox == 0 and v_By_egoy == 0):
            # speed is zero, no collisions! (also calculating the collision point would give division by zero
            #ECCCALLA', forse è questo 1 dei problemi... proprio crit 0 non credo sia giustissimo
            # --> Secondo me é ok con questa nuova modifica, ma verifichiamo
            crit_r = 0.0
            crit_t = 0.0
            print("detection/data_classes.py line 448, speed 0, no collisions -- check if OK")

        else:

            # find point of shortest distance between ego and the line given by the direction of object 
            C_x : float = ( (v_By_egoy**2)*B_X + (v_Bx_egox**2)*ego_x - v_Bx_egox*v_By_egoy*B_Y + v_Bx_egox*v_By_egoy*ego_y ) / (v_Bx_egox**2 + v_By_egoy**2)
            C_y : float = ((v_Bx_egox**2)*B_Y + (v_By_egoy**2)*ego_y + v_Bx_egox*v_By_egoy*(ego_x-B_X)) / (v_Bx_egox**2 + v_By_egoy**2)

            #if(sample_token == "9c9f22a58fdc45f2b8a119cda3554f1f"):
                #if(self.detection_name=='car' and detection_score==-1): #se car e ground truth:
                    #print("--------------")
                    #print(B_X)
                    #print(B_Y)
                    #print("-> ego coordinates:")
                    #print(ego_coordinates)
                    #print("-> object coordinates:")
                    #print(translation)
                    #print("-> ego speed:")
                    #print(self.ego_speed)
                    #print("-> object speed:")
                    #print(self.velocity)
                    #print("-> collision point:")
                    #print(C_x)
                    #print(C_y)
               
                
            # distance between ego and the line
            d_ego = math.sqrt((ego_x-C_x)**2 + (ego_y-C_y)**2)
            # distance between object and the instersection point
            d_B = math.sqrt((B_X-C_x)**2 + (B_Y-C_y)**2)

            # here we need to check if the object is moving in the "right" direction.
            # it could happen that the intersection point is in the opposite direction and thus
            # the object would never reach it
            deltax = (C_x - B_X)
            deltay = (C_y - B_Y)
            correct_direction: bool = True
            if (deltax > 0 and v_Bx_egox < 0) or (deltax < 0 and v_Bx_egox > 0):
            # the object is moving in the opposite direction on x axis
                correct_direction = False
            if (deltay > 0 and v_By_egoy < 0) or (deltay < 0 and v_By_egoy > 0):
                # the object is moving in the opposite direction on y axis
                correct_direction = False

            if correct_direction:
                # time to reach the intersection point is the distance divided by the speed
                v = math.sqrt(v_Bx_egox**2 + v_By_egoy**2)
                t_intersect_B = d_B/v #TODO: anche questo può essere NaN --- in realtà no, perché v!=0
            else:
                # object is moving in the opposite direction, time to reach the point is infinite
                t_intersect_B = math.inf
                d_ego = math.inf
                
            if(math.isnan(t_intersect_B)):
                t_intersect_B=0.1
                print("data_classes.py: NaN TROVATO nel calcolo di t_intersect_B!")
                

            # criticality wrt to distance of ego from intersection point
            # calculated as a 2nd grade equation (parabola) passing from (0,1) and (MAX_DISTANCE_INTERSECT,0)
            # the value is 1.0 when r=0 and decreases until 0.0 when r=MAX_DISTANCE_INTERSECT
            crit_r = -((d_ego**2) / (self.MAX_DISTANCE_INTERSECT**2)) + 1
            crit_r = (crit_r, 0)[crit_r < 0]    # set to zero if negative
#            print('crit_r '+ str(crit_r));
            # criticality wrt to time for object to reach the intersection point
            # calculated as a 2nd grade equation (parabola) passing from (0,1) and (MAX_TIME_INTERSECT_OBJ,0)
            # the value is 1.0 when r=0 and decreases until 0.0 when t=MAX_TIME_INTERSECT_OBJ
            crit_t = -((t_intersect_B**2) / (self.MAX_TIME_INTERSECT_OBJ**2)) + 1
            crit_t = (crit_t, 0)[crit_t < 0]    # set to zero if negative
#            print('crit_t '+str(crit_t));

        # Qui torniamo fuori dall'IF, questa parte é uguale per tutti i casi
        self.crit_t=crit_t
        self.crit_r=crit_r
        self.crit_d=crit_d

        # the final criticality is obtained by combination of the thee values
        # is 0.0 if all are zero, and 1.0 if at least one is 1.0
        self.crit = 1-(1-crit_t)*(1-crit_r)*(1-crit_d)
#       print('self.crit '+str(self.crit));

        if(math.isnan(self.crit) or math.isnan(self.crit_t) or math.isnan(self.crit_r) or math.isnan(self.crit_d)):
            print("data_classes.py: NaN TROVATO nel calcolo dei vari crit!")
           
#        if(sample_token == "9c9f22a58fdc45f2b8a119cda3554f1f" and detection_score==-1 and self.detection_name=='car'):
            #se car e ground truth:
#            print('crit_r '+str(crit_r))

    def __eq__(self, other):
        return (self.sample_token == other.sample_token and
                self.translation == other.translation and
                self.size == other.size and
                self.rotation == other.rotation and
                self.velocity == other.velocity and
                self.ego_translation == other.ego_translation and
                self.num_pts == other.num_pts and
                self.detection_name == other.detection_name and
                self.detection_score == other.detection_score and
                self.attribute_name == other.attribute_name)

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'sample_token': self.sample_token,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'ego_translation': self.ego_translation,
            'num_pts': self.num_pts,
            'detection_name': self.detection_name,
            'detection_score': self.detection_score,
            'attribute_name': self.attribute_name,
            'ego_speed': self.ego_speed
        }

    @classmethod
    def deserialize(cls, content: dict, nusc):
        """ Initialize from serialized content. """
        #print(type(cls))
        return cls(sample_token=content['sample_token'],
                   translation=tuple(content['translation']),
                   size=tuple(content['size']),
                   rotation=tuple(content['rotation']),
                   velocity=tuple(content['velocity']),
                   ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                   else tuple(content['ego_translation']),
                   num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                   detection_name=content['detection_name'],
                   detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                   attribute_name=content['attribute_name'],
                   nusc=nusc #TODO SARA QUESTO?
                  )


    def speed(self, coord_a, time_a, coord_b, time_b): #speed: A is the point forward in time
        position_diff=[0.0, 0.0]
        time_diff=float(time_a - time_b)
        position_diff[0]=float(coord_a[0])-float(coord_b[0])
        position_diff[1]=float(coord_a[1])-float(coord_b[1])

        speed_x=((position_diff[0]*1000000.0) / time_diff) # meter/microseconds *1000000 --> m/s
        speed_y=((position_diff[1]*1000000.0) / time_diff)
        #speed_z=(position_diff[2]*1000000.0/time_diff) #always 0
        speed_total=math.sqrt(speed_x**2 +speed_y**2) # +speed_z**2)--> z always 0
        return [speed_x, speed_y, speed_total]


class DetectionMetricDataList:
    """ This stores a set of MetricData in a dict indexed by (name, match-distance). """

    def __init__(self):
        self.md = {}

    def __getitem__(self, key):
        return self.md[key]

    def __eq__(self, other):
        eq = True
        for key in self.md.keys():
            eq = eq and self[key] == other[key]
        return eq

    def get_class_data(self, detection_name: str) -> List[Tuple[DetectionMetricData, float]]:
        """ Get all the MetricData entries for a certain detection_name. """
        return [(md, dist_th) for (name, dist_th), md in self.md.items() if name == detection_name]

    def get_dist_data(self, dist_th: float) -> List[Tuple[DetectionMetricData, str]]:
        """ Get all the MetricData entries for a certain match_distance. """
        return [(md, detection_name) for (detection_name, dist), md in self.md.items() if dist == dist_th]

    def set(self, detection_name: str, match_distance: float, data: DetectionMetricData):
        """ Sets the MetricData entry for a certain detection_name and match_distance. """
        self.md[(detection_name, match_distance)] = data

    def serialize(self) -> dict:
        return {key[0] + ':' + str(key[1]): value.serialize() for key, value in self.md.items()}

    @classmethod
    def deserialize(cls, content: dict, nusc):
        mdl = cls()
        for key, md in content.items():
            name, distance = key.split(':')
            mdl.set(name, float(distance), DetectionMetricData.deserialize(md))
        return mdl
