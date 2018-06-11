#! /usr/bin/env python
# encoding: utf-8
# vim: set tabstop=4:
# vim: set softtabstop=4:
# vim: set shiftwidth=4:
# vim: set expandtab:
from __future__ import print_function

__author__ = 'Benjamin Milde'

import os
import os.path
import flask
import datetime
import kaldi_io
import argparse

from flask import Flask,jsonify,json,Response
#from werkzeug.serving import WSGIRequestHandler

app = Flask(__name__)

open_sessions = {}

def get_avail_reps(feat_dir='./feats/'):
    for dirpath, dirnames, filenames in os.walk(feat_dir):
        for filename in [f for f in filenames if f.endswith(".ark")]:
            yield dirpath + '/' + filename

# enumerate the feats directory
avail_reps = {} 

# example curl usage:
# curl localhost:5000/list_avail_reps
@app.route('/list_avail_reps', methods=['GET'])
def list_avail_reps():
    response_str = json.dumps({'status':'success', 'avail_reps':avail_reps})
    response = Response(response_str,  mimetype='application/json')
    return response

# example curl usage:
# curl localhost:5000/start_session
@app.route('/start_session', methods=['GET'])
def start_session():
    # Get session ID. See http://stackoverflow.com/questions/817882/unique-session-id-in-python/6092448#6092448
    sessionid = base64.urlsafe_b64encode(M2Crypto.m2.rand_bytes(session_bytes))
    open_sessions[sessionid] = datetime.datetime.now()
    print('[startSession] Created new sessionid',sessionid)
    return sessionid

# example curl usage:
# curl localhost:5000/get_vectors -d "feat_file=./feats/tedlium_ivectors_sp/test/ivector_online.ark&half_index=-1&average_utts=True"
@app.route('/get_vectors', methods=['POST'])
def get_vectors():
    # possible parameters
    # feat_file -> path to feat_file, must be one returned by /list_avail_reps
    
    # half_index -> cut vectors at this position (optional, default: -1)
    # limit -> max vectors to return (optional, numeric)
    # average_utts -> average vector for each utterance (optional, default: True)
    # normalize -> normalize vectors to unit length (optional, default: False)
   
    # Reading parameters from POST request:
    if 'feat_file' in flask.request.form:
        feat_filename = flask.request.form['feat_file']
    else:
        response_str = json.dumps({'status':'fail', 'reason':'You must supply a feat_file for /get_vectors'})
        response = Response(response_str,  mimetype='application/json')
        return response
    
    if 'half_index' in flask.request.form:
        half_index = int(flask.request.form['half_index'])
    else:
        print('POST /get_vectors called without half_index parameter, setting to default -1 (disable)')
        half_index = -1
        
    if 'limit' in flask.request.form:
        limit = int(flask.request.form['limit'])
    else:
        print('POST /get_vectors called without limit parameter, setting to default -1 (disable)')
        limit = -1
        
    if 'average_utts' in flask.request.form:
        average_utts = flask.request.form['average_utts']
    else:
        print('POST /get_vectors called without average_utts parameter, setting to default true (enable)')
        average_utts = True
        
    normalize = ('normalize' in flask.request.form)

    feats, utt_ids = kaldi_io.readArk(feat_filename, limit=limit)
    
    feats_len=len(feats)
    
    assert(len(utt_ids)==len(feats))
    
    print("Loaded:" + str(feats_len) + " feats.")

    if average_utts or average_utts == 'True' or average_utts == 'true':    
        feats = [feat.mean(0) for feat in feats]
        
        if half_index != -1:
            print('Cutting vectors at ', half_index, 'and normalize to unit length' if normalize else '')
            feats = [feat[:half_index]/(np.linalg.norm(feat[:half_index]) if normalize else 1.0) for feat in feats]
        else:
            if normalize:
                print('Normalize to unit length.')
                feats = [feat/np.linalg.norm(feat) for feat in feats]
        
        response_vec_dict = {}
        
        for utt_id, feat in zip(utt_ids, feats):
            response_vec_dict[utt_id] = feat.tolist()
        
        response_str = json.dumps({'status':'success', 'vectors':response_vec_dict})
        
    else:
        print('Not yet supported')
        response_str = json.dumps({'status':'fail', 'reason':'average_utts==False not yet supported'})
    
    response = Response(response_str,  mimetype='application/json')         
    return response

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Unspeech feature access server (used for HTML tsne vizualisation)')
    parser.add_argument('-l', '--listen-host', default='127.0.0.1', dest='host', help='Host address to listen on.', type=str)
    parser.add_argument('-p', '--port', dest='port', default=5000, help='Port to listen on.', type=int)
    parser.add_argument('-f', '--feat-dir', dest='feat_dir', default='./feats/', help='Path to the feature data directory.', type=str)
    parser.add_argument('--test-server', dest='test_server', help='Start the flask intern web server.', action='store_true', default=True)

    args = parser.parse_args()

    avail_reps = list(get_avail_reps(feat_dir=args.feat_dir))
    
    #print('Available representations:')
    #print(avail_reps)
    
    def run_twisted_wsgi():
        from twisted.internet import reactor
        from twisted.web.server import Site
        from twisted.web.wsgi import WSGIResource

        resource = WSGIResource(reactor, reactor.getThreadPool(), app)
        site = Site(resource)
        reactor.listenTCP(args.port, site)
        reactor.run(**reactor_args)

    if args.test_server:
        print('Running as testing server.')
        print('Host:', args.host, 'port:', args.port)
        app.debug = True
        app.run(host=args.host, port=args.port, threaded=False)
    else:
        print('Running as deployment server.')
        print('Host:', args.host, 'port:', args.port)
        run_twisted_wsgi()