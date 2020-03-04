#!/bin/bash

set -e

curl -fLo ./symmetric.pkl https://neuralalignpkls.s3-us-west-1.amazonaws.com/symmetric.pkl
curl -fLo ./activation.pkl https://neuralalignpkls.s3-us-west-1.amazonaws.com/activation.pkl
curl -fLo ./wm.pkl https://neuralalignpkls.s3-us-west-1.amazonaws.com/wm.pkl
curl -fLo ./wm_tpe.pkl https://neuralalignpkls.s3-us-west-1.amazonaws.com/wm_tpe.pkl
curl -fLo ./wm_tpe_ad.pkl https://neuralalignpkls.s3-us-west-1.amazonaws.com/wm_tpe_ad.pkl
curl -fLo ./wm_tpe_ad_ops.pkl https://neuralalignpkls.s3-us-west-1.amazonaws.com/wm_tpe_ad_ops.pkl
curl -fLo ./information_tpe.pkl https://neuralalignpkls.s3-us-west-1.amazonaws.com/information_tpe.pkl
