import React from "react";
import styled from "@emotion/styled";

import mediaqueries from "@styles/media";

import { Icon } from '@types';

const Logo: Icon = ({ fill = "white" }) => {
  return (
    <LogoContainer>
      <svg
        width="245"
        height="49"
        viewBox="0 0 245 49"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="Logo__Desktop"
      >
        <g clipPath="url(#clip0)">
          <path
       d="m 194.35629,24.218512 -0.19388,0.09694 q 0.14541,0.14541 0.19388,0.14541 0.0485,0 0.0485,-0.04847 0.0485,-0.09694 0.14541,-0.14541 0.0969,-0.09694 0.29082,-0.09694 1.89032,0.14541 3.15054,0 1.30869,-0.193879 2.37502,-0.630108 1.11481,-0.436229 2.22962,-1.066337 1.1148,-0.678578 2.71431,-1.502566 l 1.40563,-0.678578 q 0.38775,-0.09694 0.6301,-0.193879 0.29082,-0.09694 0.63011,-0.09694 0.0485,0.387759 -0.29082,0.630108 -0.33928,0.242349 -0.33928,0.630108 l -0.72705,0.387759 q -0.0969,0.04847 0,0.09694 0.0969,0 0.0969,0.09694 -0.67858,0.339289 -1.21175,0.678578 -0.53317,0.290819 -1.11481,0.678578 -0.38776,0.290819 -0.82398,0.484699 -0.38776,0.145409 -0.63011,0.484698 -1.26022,0.436229 -2.08421,0.872458 -0.77552,0.387759 -1.50256,0.678578 -0.72705,0.24235 -1.59951,0.387759 -0.87246,0.14541 -2.32655,0.04847 0.67858,0.872458 1.01786,1.938795 0.38776,1.066337 0.53317,2.181144 0.19388,1.114807 0.14541,2.132674 0,1.066337 -0.14541,1.841855 0,0.09694 0.0969,0.04847 0.0969,0 0.0485,0.145409 -0.29082,1.647976 -0.77551,3.247481 -0.4847,1.647976 -1.16328,3.247482 -0.38776,0.727048 -0.77552,1.502566 -0.38776,0.775517 -0.87246,1.454096 -0.58163,0.727048 -1.21174,1.357156 -0.38776,0.484699 -0.72705,0.823988 -0.33929,0.387759 -0.82399,0.823987 -0.19388,0.09694 -0.29082,0.24235 -0.0969,0.193879 -0.4847,0.290819 -0.0969,0.04847 -0.24235,0 -0.14541,0 -0.24235,0.04847 -0.33928,0.09694 -0.53316,0.242349 -0.24235,0.14541 -0.63011,0.242349 -0.38776,0.09694 -0.72705,-0.09694 -0.38776,-0.14541 -0.72705,-0.290819 -0.19388,-0.09694 -0.33929,-0.19388 -0.19388,-0.04847 -0.29082,-0.14541 -0.33929,-0.290819 -0.58163,-0.920927 -0.29082,-0.581638 -0.43623,-1.066337 -0.29082,-1.357156 -0.24235,-2.762783 0.0485,-1.357156 0.29082,-2.617372 0.24235,-1.841855 0.77551,-3.97453 0.53317,-2.132674 1.21175,-4.313818 0.67858,-2.132674 1.40563,-4.168408 0.67858,-2.084205 1.26021,-3.73218 l 0.14541,0.145409 0.0969,-0.387759 0.19388,-0.387759 0.19388,-0.823987 0.58164,-1.017868 0.19388,-1.017867 q 0.33929,-0.581638 0.58164,-1.163277 0.29082,-0.581638 0.53317,-1.163276 0.0969,0 0.0485,-0.04847 0,-0.09694 0.0969,-0.04847 -0.0485,-0.14541 0,-0.290819 0.0969,-0.14541 0.19388,-0.24235 l 0.14541,-0.484698 q 0.1454,-0.29082 0.33928,-0.533169 0.19388,-0.290819 0.33929,-0.581638 0.14541,-0.387759 0.33929,-0.775518 0.24235,-0.387759 0.53317,-0.775518 0.0969,0 0.0485,-0.04847 0,-0.04847 0.0485,-0.14541 0.0969,-0.290819 0.29082,-0.630108 0.24235,-0.339289 0.43623,-0.581638 0.14541,-0.290819 0.29082,-0.630109 0.14541,-0.339289 0.4847,-0.533168 0.38776,-0.823988 0.87246,-1.502566 0.48469,-0.6785778 1.01786,-1.3571558 l 0.87246,-1.357157 q 0,-0.04847 0.0485,-0.04847 0.0969,0 0.14541,-0.04847 l 0.33929,-0.533169 q 0.33929,-0.290819 0.58164,-0.581638 0.24235,-0.339289 0.53317,-0.581639 0,-0.09694 0.0485,-0.09694 0.0969,0 0,-0.09694 0.43622,-0.436229 0.77551,-0.823988 0.38776,-0.387759 0.67858,-0.823988 0.43623,-0.2908188 0.87246,-0.7270476 0.43623,-0.4846987 1.26022,-0.387759 0.0969,-0.04847 0.29081,-0.04847 0.19388,-0.04847 0.33929,-0.1938794 l 2.22962,0.5816384 q 0.0969,0.1938795 0.29082,0.4362288 0.24235,0.1938795 0.38776,0.4362288 0.38776,0.533169 0.0485,1.163277 -0.33929,0.581638 -0.38776,1.114807 -0.0969,0.09694 -0.0969,0.193879 0,0.04847 -0.14541,0.04847 -0.0485,0.19388 -0.0485,0.24235 -0.38776,0.630108 -0.72705,1.260216 -0.33929,0.630108 -0.72705,1.308687 -0.0969,0 -0.0969,0.09694 0.0485,0.04847 0,0.04847 0,0.09694 -0.0485,0.14541 -0.0485,0 0.0485,0.09694 -0.0485,0.04847 -0.0969,0.04847 h -0.0485 q 0,0.19388 -0.14541,0.29082 -0.14541,0.09694 -0.19388,0.3392888 -0.58164,0.920927 -1.11481,1.841855 -0.4847,0.920927 -1.26021,1.744915 -0.24235,0.533168 -0.63011,1.017867 -0.33929,0.436229 -0.72705,0.920928 -0.0969,0.242349 -0.24235,0.533168 -0.0969,0.290819 -0.33929,0.387759 0.0485,0.14541 -0.0969,0.242349 -0.14541,0.09694 -0.14541,0.19388 -0.0969,0.09694 -0.14541,0.145409 0,0.04847 -0.0969,0.09694 -0.67857,0.920928 -1.35715,1.890325 -0.63011,0.969397 -1.35716,1.938795 -0.67858,0.920927 -1.50257,1.744915 -0.82398,0.823988 -1.89032,1.405626 0.0485,0 0.0969,0 0.0485,-0.04847 0.14541,-0.04847 z m -4.023,6.349553 q -0.29082,0.533168 -0.4847,1.066337 -0.14541,0.581638 -0.29082,1.211747 -0.53317,1.744915 -1.01786,3.39289 -0.4847,1.696446 -0.77552,3.635241 -0.14541,1.211746 -0.19388,2.520433 -0.0969,1.308686 0.33929,2.714312 0.38776,1.163277 1.06633,1.308687 1.01787,-0.872458 1.69645,-1.938795 0.72705,-1.017867 1.26022,-2.181144 1.69644,-3.344421 2.5689,-7.076601 0.24235,-1.163277 0.33929,-2.617373 0.14541,-1.405626 -0.29082,-2.811252 -0.24235,-0.969397 -0.67858,-1.599506 -0.43623,-0.630108 -0.87246,-1.260216 -0.24235,-0.242349 -0.4847,-0.533169 -0.19387,-0.290819 -0.58163,-0.193879 -0.53317,0.969397 -0.82399,2.132674 -0.29082,1.114807 -0.77552,2.229614 z m 3.24748,-7.852119 q 2.76278,-3.34442 4.9924,-6.640371 2.27808,-3.295952 4.45923,-7.0766008 0.38775,-1.114807 0.96939,-2.132674 0.63011,-1.066337 0.87246,-2.375024 -0.0969,-0.04847 -0.19388,-0.04847 -0.0969,0 -0.14541,0 -0.82399,0.727048 -1.69644,1.551036 -0.87246,0.775518 -1.4541,1.696445 h -0.0969 q -0.0485,0.09694 -0.38776,0.630109 -0.29082,0.533168 -0.67858,1.163276 -0.38776,0.6301088 -0.77551,1.2117468 -0.33929,0.533169 -0.4847,0.678578 0,0.19388 -0.33929,0.436229 0,0.09694 -0.19388,0.484699 -0.14541,0.387759 -0.38776,0.823988 -0.19388,0.387759 -0.43623,0.775517 -0.19388,0.387759 -0.24235,0.484699 -0.0969,0.242349 -0.29082,0.484699 -0.14541,0.242349 -0.33929,0.339289 l 0.14541,0.09694 q -0.14541,0 -0.24235,0.193879 -0.0969,0.19388 -0.14541,0.436229 -0.0485,0.242349 -0.29082,0.339289 0.14541,0.193879 -0.0485,0.290819 -0.19387,0.04847 -0.0969,0.24235 h -0.1454 q 0.19387,0.145409 -0.0485,0.290819 -0.19388,0.145409 -0.14541,0.242349 -0.0485,0 -0.0969,0.04847 0,0 -0.0485,0 -0.38776,0.920927 -0.77552,1.841855 -0.38776,0.920927 -0.72705,1.696445 -0.29082,0.727048 -0.43623,1.211747 -0.14541,0.484699 -0.0485,0.581638 z m -3.5383,0.775518 q -0.19388,-0.09694 -0.24235,0 0,0.04847 0.14541,0.19388 z"
            fill="#7A8085"
          />
          <path
       d="m 208.07319,26.399656 q 0.14541,0.14541 0.82398,-0.09694 0.67858,-0.29082 1.64798,-0.727048 1.01787,-0.484699 2.22961,-1.066338 1.21175,-0.581638 2.4235,-1.017867 0.33929,-0.09694 0.77551,-0.290819 0.4847,-0.193879 0.72705,-0.339289 0.38776,-0.242349 0.77552,-0.484699 0.38776,-0.242349 0.92093,-0.339289 0.19388,-0.04847 0.19388,-0.145409 0,-0.14541 0.0485,0.04847 0.0969,0.04847 0.14541,0.04847 0.0485,-0.04847 0.0969,0 0,0.19388 -0.24235,0.533169 l -0.33929,0.290819 q -0.38776,0.339289 -0.82399,0.678578 -0.43623,0.290819 -0.9694,0.533169 -0.43623,0.387759 -1.06633,0.727048 -0.58164,0.339289 -1.16328,0.630108 -0.53317,0.19388 -1.55104,0.872458 -0.96939,0.678578 -2.0842,1.405626 -1.06634,0.727048 -1.98726,1.308686 -0.92093,0.533169 -1.26022,0.387759 -0.14541,0.14541 -0.19388,0.09694 -0.0485,0 -0.19388,0 -0.77552,0 -1.35716,-0.14541 -0.53316,-0.193879 -0.92092,-0.727048 -0.14541,-0.581638 0,-1.163276 0.14541,-0.581639 0.38776,-1.308687 0.1454,-0.387759 0.29081,-0.727048 0.19388,-0.339289 0.29082,-0.727048 0.19388,-0.387759 0.38776,-0.823988 0.19388,-0.436228 0.33929,-0.823987 0.53317,-1.114807 1.11481,-2.229614 0.58164,-1.163277 1.26021,-2.132675 0.38776,-0.242349 0.72705,-0.290819 0.19388,-0.09694 0.38776,0.04847 0.24235,0.09694 0.38776,0 0.14541,0.19388 0.29082,0.339289 0.19388,0.14541 0.24235,0.387759 0.0485,0.339289 0.0485,0.533169 0.0485,0.193879 -0.0485,0.436229 0.0485,0.339289 -0.19388,0.727048 -0.0969,0.145409 -0.14541,0.339289 -0.0485,0.145409 -0.14541,0.339289 -0.0485,0.04847 -0.0969,0.04847 0,-0.04847 -0.0485,0 -0.0969,0.04847 -0.0969,0.145409 0,0.09694 -0.0969,0.14541 -0.33929,0.678578 -0.72705,1.405626 -0.38776,0.727048 -0.72705,1.405626 -0.33928,0.630108 -0.48469,1.114807 -0.14541,0.484699 0,0.630108 z m 1.98726,-11.826647 q -0.0485,-0.14541 0.24235,-0.533169 0.29082,-0.387759 0.67858,-0.727048 0.38776,-0.339289 0.77552,-0.484699 0.38775,-0.145409 0.6301,0.14541 0.0969,0 0.14541,0.04847 0.0969,0.04847 0.14541,0.04847 0.33929,0.387759 0.19388,0.969397 -0.0969,0.533169 -0.4847,0.727048 0.0485,0.09694 0,0.14541 -0.0485,0 -0.0485,0.09694 -0.14541,-0.09694 -0.14541,0.09694 0.0485,0.19388 -0.14541,0.09694 0.0485,0.145409 -0.14541,0.290819 -0.1454,0.14541 -0.1454,0.290819 -0.0969,-0.04847 -0.0969,0 0,0 -0.0969,0 -0.0969,0.24235 -0.43623,0.387759 -0.29082,0.09694 -0.43623,0.290819 -0.0969,0 -0.19388,0.04847 -0.0485,0.04847 -0.24235,-0.04847 -0.29082,-0.09694 -0.38776,-0.290819 -0.0969,-0.242349 -0.0969,-0.533168 0.0485,-0.29082 0.14541,-0.581639 0.0969,-0.290819 0.14541,-0.484699 z"
            fill="#7A8085"
          />
          <path
       d="m 228.91522,17.917429 q -0.0969,0.14541 -0.0969,0.24235 0.0485,0.09694 -0.14541,0.193879 0,0.09694 -0.14541,0.09694 -0.0969,0.242349 -0.38776,0.290819 -0.24235,0.04847 -0.43623,0.290819 -0.96939,0.29082 -2.03573,0.727048 -1.06634,0.436229 -2.08421,0.969398 -0.96939,0.484698 -1.79338,1.017867 -0.82399,0.533169 -1.26022,1.066337 -0.38776,0.387759 -0.82399,0.823988 -0.43622,0.436229 -0.77551,0.969397 -0.14541,0.387759 -0.33929,0.969398 -0.14541,0.581638 -0.29082,1.163276 -0.14541,0.533169 -0.33929,0.969398 -0.14541,0.436229 -0.29082,0.533168 -0.0969,0.19388 -0.38776,0.533169 -0.29082,0.339289 -0.4847,0.290819 -0.72704,0.339289 -1.26021,-0.09694 -0.4847,-0.436228 -0.33929,-1.211746 0.0969,-0.484699 0.43623,-1.454096 0.33929,-0.969398 0.43623,-1.502566 0,0 0,-0.04847 0.0485,-0.14541 0.0485,-0.290819 0.38776,-0.678579 0.63011,-1.405627 0.24235,-0.727048 0.58164,-1.405626 0,-0.04847 0.0485,-0.04847 0.0969,-0.04847 0.0969,-0.09694 0.24235,-0.969398 0.58164,-1.890325 0.38776,-0.969398 0.67858,-1.841855 0.0485,-0.19388 0.14541,-0.290819 0.0969,-0.09694 0.19388,-0.24235 0.24235,-0.09694 0.38775,-0.193879 0.19388,-0.09694 0.38776,-0.09694 0.0969,-0.09694 0.24235,-0.09694 0.0969,0 0.19388,0.04847 0.14541,0 0.24235,0 0.43623,0.04847 0.63011,0.339289 0.58164,0.775518 -0.0969,1.454096 -0.0485,0.387759 -0.19388,0.775518 -0.0969,0.387759 -0.14541,0.678578 1.69645,-1.260216 3.34442,-1.744915 1.64798,-0.484699 3.0536,-0.581638 0.38776,-0.04847 0.87246,-0.09694 0.53317,-0.04847 0.92093,0.193879 z"
            fill="#7A8085"
          />
          <path
       d="m 233.7622,27.853752 q -0.38776,0.24235 -0.87246,0.387759 -0.43623,0.09694 -0.87246,0.24235 -0.58163,0.145409 -1.30868,0.290819 -0.72705,0.145409 -1.4541,0.145409 -0.72705,-0.04847 -1.40562,-0.290819 -0.63011,-0.290819 -1.06634,-0.920927 -0.33929,-0.436229 -0.38776,-1.066337 -0.0485,-0.630109 0.0969,-1.260217 0.14541,-0.678578 0.38776,-1.308686 0.29082,-0.678578 0.58164,-1.211747 0.24235,-0.193879 0.38776,-0.484699 0.14541,-0.290819 0.19388,-0.484698 1.01786,-1.211747 2.0842,-2.326554 1.11481,-1.114807 2.27808,-1.744915 0.53317,-0.339289 1.11481,-0.581639 0.58164,-0.242349 1.06634,-0.484698 0.19388,-0.09694 0.29082,-0.19388 0.14541,-0.09694 0.38776,-0.09694 1.74491,0 2.27808,0.630109 0.53317,0.581638 0.53317,1.163277 -0.0485,0.193879 -0.33929,0.436228 -0.29082,0.24235 -0.43623,0.484699 -0.24235,0.242349 -0.43623,0.484699 -0.19388,0.193879 -0.29082,0.387759 -1.93879,1.696445 -3.87759,3.005131 -0.4847,0.387759 -1.30868,0.533169 -0.77552,0.14541 -1.59951,0.04847 -0.29082,0.290819 -0.58164,0.775518 -0.29082,0.436229 -0.4847,0.969397 -0.14541,0.484699 -0.14541,1.017867 0.0485,0.484699 0.43623,0.823988 0.9694,0.09694 1.69645,-0.193879 0.72705,-0.29082 1.40562,-0.484699 0.53317,-0.14541 1.11481,-0.339289 0.58164,-0.19388 1.01787,-0.387759 1.01787,-0.484699 1.64797,-0.775518 0.67858,-0.339289 1.26022,-0.630108 0.58164,-0.290819 1.21175,-0.630109 0.63011,-0.339289 1.5995,-0.872457 0.77552,-0.387759 1.55104,-0.775518 0.82399,-0.436229 1.55103,-0.823988 0.29082,-0.242349 0.53317,-0.339289 0.24235,-0.145409 0.63011,-0.242349 0.0485,0 0.0969,-0.04847 0.0969,-0.04847 0.14541,-0.04847 0.43623,0 0.43623,0.14541 0,0.145409 -0.24235,0.436228 -0.19388,0.24235 -0.53317,0.533169 -0.29082,0.290819 -0.4847,0.581638 0.0485,0 -0.14541,0.14541 -0.14541,0.09694 -0.38776,0.242349 -0.19388,0.09694 -0.38775,0.24235 -0.14541,0.09694 -0.14541,0.09694 -0.53317,0.387759 -1.30869,0.872458 -0.77552,0.436229 -1.55104,0.920928 -0.77551,0.436228 -1.45409,0.823987 -0.67858,0.387759 -1.01787,0.533169 z m -3.48983,-4.992396 q 0.43623,0 0.82399,-0.242349 0.43623,-0.24235 0.82398,-0.484699 0.9694,-0.775518 1.98727,-1.696445 1.01787,-0.969398 1.84185,-1.938795 0.0969,0 0.0969,-0.14541 -1.74491,0.630109 -3.10207,1.841855 -1.30868,1.211747 -2.47196,2.665843 z"
            fill="#7A8085"
          />
          <path
            d="m 0,14.423613 h 10.09653 v 24.52014 H 0 Z"
            fill={fill}
          />
          <path
            d="M 14.42361,-5.4749352e-7 H 24.52014 V 38.943744 H 14.42361 Z"
            fill={fill}
          />
          <path
            d="M 28.847226,-5.4749352e-7 H 38.943752 V 24.520139 H 28.847226 Z"
            fill={fill}
          />
          <path
       d="m 54.480625,25.331541 -1.704389,5.43855 h -2.64955 L 57.068197,9.4342412 h 3.439767 l 6.941511,21.3358498 h -2.819989 l -1.704389,-5.43855 z m 7.747222,-2.2312 -3.501744,-11.403911 -3.548228,11.403911 z"
            fill={fill}
          />
          <path
       d="m 77.551808,26.741535 q 0,1.146589 0.712744,1.688895 0.728239,0.526811 1.983289,0.526811 1.317028,0 2.742517,-0.573295 l 0.69725,1.905817 q -0.743733,0.371867 -1.735378,0.604283 -0.991644,0.232417 -2.138233,0.232417 -1.440984,0 -2.556584,-0.542305 -1.100105,-0.542306 -1.704388,-1.564939 -0.604284,-1.022634 -0.604284,-2.417134 V 9.9455582 h -5.004705 v -2.060762 h 7.607772 z"
            fill={fill}
          />
          <path
       d="m 103.86133,15.167185 q -0.96066,0.294395 -2.06076,0.387361 -1.08461,0.09297 -2.711529,0.09297 2.835479,1.301533 2.835479,4.106028 0,1.611422 -0.74373,2.881967 -0.72824,1.25505 -2.122738,1.967794 -1.379006,0.69725 -3.269328,0.69725 -0.805711,0 -1.363511,-0.06198 -0.542306,-0.07747 -1.069117,-0.247911 -0.371866,0.263406 -0.619778,0.69725 -0.232416,0.41835 -0.232416,0.883183 0,0.635273 0.495822,1.007139 0.495822,0.356373 1.688894,0.356373 h 2.943945 q 1.611422,0 2.928447,0.588788 1.33253,0.573295 2.07626,1.595928 0.75923,1.007139 0.75923,2.246695 0,2.401638 -2.02978,3.687677 -2.029768,1.286039 -5.763929,1.286039 -2.572078,0 -4.090534,-0.526811 -1.502961,-0.526811 -2.153727,-1.611422 -0.650767,-1.069117 -0.650767,-2.773506 h 2.355155 q 0,1.007139 0.371867,1.595928 0.387361,0.604283 1.363511,0.898678 0.991645,0.294394 2.789,0.294394 2.665045,0 3.920095,-0.666261 1.255049,-0.666261 1.255049,-2.014277 0,-1.131095 -1.022633,-1.735378 -1.022633,-0.604284 -2.665044,-0.604284 h -2.912956 q -1.348016,0 -2.277683,-0.41835 -0.929667,-0.41835 -1.379006,-1.1156 -0.449338,-0.69725 -0.449338,-1.53395 0,-0.790216 0.464833,-1.53395 0.464833,-0.743733 1.332522,-1.317027 -1.409994,-0.743734 -2.076255,-1.843839 -0.666262,-1.100106 -0.666262,-2.665045 0,-1.626916 0.821206,-2.912955 0.8367,-1.301534 2.308672,-2.029772 1.487467,-0.728239 3.377789,-0.728239 1.967795,0 3.253833,-0.154945 1.286042,-0.170439 2.153732,-0.449339 0.88318,-0.2789 2.02977,-0.774722 z m -8.243045,0.8367 q -1.905817,0 -2.881967,1.022634 -0.960655,1.022633 -0.960655,2.742516 0,1.735378 0.97615,2.773506 0.991644,1.022633 2.92845,1.022633 1.704389,0 2.618561,-0.991644 0.929667,-1.007139 0.929667,-2.819989 0,-1.843839 -0.914173,-2.789 -0.914172,-0.960656 -2.696033,-0.960656 z"
            fill={fill}
          />
          <path
       d="m 110.43093,23.534185 q 0.062,1.828345 0.71274,3.052406 0.65077,1.208567 1.70439,1.797355 1.05362,0.573295 2.35516,0.573295 1.20856,0 2.2157,-0.356372 1.02264,-0.356373 2.13824,-1.1156 l 1.22406,1.719883 q -1.14659,0.898678 -2.63406,1.409994 -1.47197,0.511317 -2.97493,0.511317 -2.33966,0 -4.02856,-1.053622 -1.68889,-1.053622 -2.57207,-2.974933 -0.86769,-1.921312 -0.86769,-4.4624 0,-2.463617 0.88318,-4.400423 0.88318,-1.936805 2.49461,-3.021416 1.61142,-1.100106 3.74965,-1.100106 2.04527,0 3.53273,0.960656 1.50297,0.960655 2.29318,2.773505 0.80571,1.797356 0.80571,4.260972 0,0.712745 -0.062,1.425489 z m 4.43141,-7.282389 q -1.89032,0 -3.08339,1.332523 -1.17758,1.317027 -1.33253,3.935589 h 8.50645 q -0.0465,-2.572078 -1.13109,-3.920095 -1.08461,-1.348017 -2.95944,-1.348017 z"
            fill={fill}
          />
          <path
       d="m 129.58202,16.57718 q 0.91417,-1.193072 2.09175,-1.828345 1.17757,-0.635272 2.55658,-0.635272 3.16087,0 4.63284,2.246695 1.48747,2.2312 1.48747,6.244261 0,2.494605 -0.74374,4.431411 -0.74373,1.921311 -2.20021,3.005922 -1.45648,1.084611 -3.51724,1.084611 -1.42549,0 -2.52559,-0.495822 -1.10011,-0.511317 -1.92131,-1.549445 l -0.18594,1.688895 h -2.27768 V 7.8847962 l 2.60307,-0.325383 z m 3.68767,12.442039 q 2.07626,0 3.16087,-1.595928 1.08461,-1.611422 1.08461,-4.834267 0,-3.176361 -0.97615,-4.772289 -0.97615,-1.611422 -2.86647,-1.611422 -1.25505,0 -2.29318,0.743733 -1.03813,0.743734 -1.79735,1.874828 v 8.0881 q 0.65076,0.991645 1.61142,1.549445 0.97615,0.5578 2.07625,0.5578 z"
            fill={fill}
          />
          <path
       d="m 156.58879,14.098069 q 0.63527,0 1.16209,0.09297 0.52681,0.07747 1.14658,0.263406 l -0.37186,5.516022 h -2.13824 v -3.470755 h -0.13945 q -1.89032,0 -3.22284,1.332522 -1.33252,1.332522 -2.15373,4.028555 v 6.910523 h 3.31581 v 1.998783 h -8.45996 v -1.998783 h 2.54109 V 16.453224 h -2.54109 v -1.998783 h 4.52437 l 0.48033,3.858117 q 0.99165,-2.122739 2.35516,-3.160867 1.379,-1.053622 3.50174,-1.053622 z"
            fill={fill}
          />
          <path
       d="m 175.84834,27.206369 q 0,0.960655 0.30989,1.425489 0.30989,0.449338 1.02263,0.666261 l -0.63527,1.828344 q -2.35516,-0.309889 -2.99043,-2.293178 -0.86769,1.131095 -2.20021,1.719884 -1.31703,0.573294 -2.91296,0.573294 -1.61142,0 -2.80449,-0.619778 -1.19307,-0.619777 -1.82834,-1.750872 -0.63528,-1.146589 -0.63528,-2.665044 0,-2.525595 1.9678,-3.873611 1.96779,-1.348017 5.68646,-1.348017 h 2.40164 V 19.50563 q 0,-1.704389 -0.99165,-2.463617 -0.99164,-0.774722 -2.89746,-0.774722 -1.84384,0 -4.24548,0.8367 l -0.68175,-1.967795 q 2.72702,-1.022633 5.33009,-1.022633 2.99043,0 4.53987,1.409995 1.56494,1.3945 1.56494,3.858116 z m -6.74008,1.936805 q 1.20856,0 2.29317,-0.604283 1.10011,-0.619778 1.82835,-1.704389 v -4.198994 h -2.35516 q -2.58757,0 -3.74965,0.883183 -1.16209,0.883183 -1.16209,2.541089 0,3.083394 3.14538,3.083394 z"
            fill={fill}
          />
        </g>
        <defs>
          <clipPath id="clip0">
            <rect width="259" height="49" fill="white" />
          </clipPath>
        </defs>
      </svg>

      <svg
        width="40"
        height="30"
        viewBox="0 0 40 30"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="Logo__Mobile"
      >
        <path
        d="m 0,14.423613 h 10.09653 v 24.52014 H 0 Z"
        fill={fill} />
        <path
          d="M 14.42361,-5.4749352e-7 H 24.52014 V 38.943744 H 14.42361 Z"
          fill={fill}
        />
        <path
          d="M 28.847226,-5.4749352e-7 H 38.943752 V 24.520139 H 28.847226 Z"
          fill={fill}
        />
      </svg>
    </LogoContainer>
  );
};

export default Logo;

const LogoContainer = styled.div`
  .Logo__Mobile {
    display: none;
  }

  ${mediaqueries.tablet`
    .Logo__Desktop {
      display: none;
    }
    
    .Logo__Mobile{
      display: block;
    }
  `}
`;
;