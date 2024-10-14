from flask import Flask, render_template, request,redirect
import jsonify
import logging
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, redirect, url_for, session
import Scripts.Data_Processor_1 as dp1
import Scripts.Script_Dimensionality_clustering as clustering
import Scripts.ML as ML
import Scripts.visual as VSUL
import pandas as pd
from io import StringIO
import os
import uuid 
import re 
from werkzeug.serving import WSGIRequestHandler
from Scripts.Spirometry_results import svc_calc,mvv_calc,fvc_calc,sm_calc,raw_data_calc
from Scripts.current_analysis_Rdata_visual import main_function_visual
from Scripts.current_analysis_PCA import main_function_pca
from Scripts.current_analysis_ML import plotter_svm

# Increase maximum header size to 64 KB (65536 bytes)
#WSGIRequestHandler.protocol_version = "HTTP/1.1"
#WSGIRequestHandler.max_http_header_size = 65536

app = Flask(__name__)
app.secret_key = os.urandom(24)

# @app.before_request
# def clear_temp_data():
#     # Clear session data if you use Flask sessions
#     session.clear()

# @app.after_request
# def modify_response(response):
#     # Set headers to prevent caching
#     response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
#     response.headers['Pragma'] = 'no-cache'
#     response.headers['Expires'] = '0'

#     # Clear cookies by setting them to an empty string and max-age to 0
#     for cookie in request.cookies:
#         response.set_cookie(cookie, '', expires=0, max_age=0)

#     return response

#logging.basicConfig(level=logging.DEBUG)
UPLOAD_FOLDER = 'uploads'  # Folder to store uploaded CSV files
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
#model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('breath1.html')

@app.route('/page2breathe')
def page2breathe():
    prediction_text_placeholder = "Please wait for processor response"
    return render_template('page2breathe.html',sample_placeholder=prediction_text_placeholder)

@app.route('/redirect_to_page2')
def redirect_to_page2():
    # Redirect to the 'page2breathe' route
    return redirect(url_for('page2breathe'))


@app.route('/page2newAnalysis')
def page2newAnalysis():
    prediction_text_placeholder = "Please wait for processor response"
    return render_template('new_analysis_landing.html',sample_placeholder=prediction_text_placeholder)

@app.route('/redirect_to_new_analysis')
def redirect_to_new_analysis():
    # Redirect to the 'page2breathe' route
    return redirect(url_for('page2newAnalysis'))


@app.route('/DMwebpage')
def DMwebpage():
    prediction_text_placeholder = "Please wait for processor response"
    return render_template('DMwebpage.html',sample_placeholder=prediction_text_placeholder)
@app.route('/redirect_to_DMwebpage')
def redirect_to_DMwebpage():
    # Redirect to the 'page2breathe' route
    return redirect(url_for('DMwebpage'))

@app.route('/NewrawVisual')
def NewrawVisual():
    prediction_text_placeholder = "Please wait for processor response"
    return render_template('current_analysis_rawDatavisual.HTML',sample_placeholder=prediction_text_placeholder)
@app.route('/redirect_to_NewrawVisual')
def redirect_to_NewrawVisual():
    # Redirect to the 'page2breathe' route
    return redirect(url_for('NewrawVisual'))


@app.route('/NewPca')
def NewPca():
    prediction_text_placeholder = "Please wait for processor response"
    return render_template('current_analysis_PCA.HTML',sample_placeholder=prediction_text_placeholder)
@app.route('/redirect_to_NewPca')
def redirect_to_NewPca():
    # Redirect to the 'page2breathe' route
    return redirect(url_for('NewPca'))

@app.route('/NewML')
def NewML():
    prediction_text_placeholder = "Please wait for processor response"
    return render_template('current_analysis_ML.HTML',sample_placeholder=prediction_text_placeholder)
@app.route('/redirect_to_NewML')
def redirect_to_NewML():
    # Redirect to the 'page2breathe' route
    return redirect(url_for('NewML'))




@app.route('/ML_webpage')
def ML_webpage():
    prediction_text_placeholder = "Please wait for processor response"
    return render_template('ML_webpage.html',sample_placeholder=prediction_text_placeholder)
@app.route('/redirect_to_ML_webpage')
def redirect_to_ML_webpage():
    # Redirect to the 'page2breathe' route
    return redirect(url_for('ML_webpage'))

@app.route('/clusterwebpage')
def clusterwebpage():
    prediction_text_placeholder = "Please wait for processor response"
    return render_template('clusterwebpage.html',sample_placeholder=prediction_text_placeholder)
@app.route('/redirect_to_clusterwebpage')
def redirect_to_clusterwebpage():
    # Redirect to the 'page2breathe' route
    return redirect(url_for('clusterwebpage'))

@app.route('/visual')
def visual():
    prediction_text_placeholder = "Please wait for processor response"
    return render_template('visual.html',sample_placeholder=prediction_text_placeholder)

@app.route('/redirect_to_visual')
def redirect_to_visual():
    # Redirect to the 'page2breathe' route
    return redirect(url_for('visual'))

@app.route('/breathepage')
def breathepage():
    prediction_text_placeholder = "Please wait for processor response"
    return render_template('BreatheAnalysisHome.html',sample_placeholder=prediction_text_placeholder)
@app.route('/redirect_to_breathepage')
def redirect_to_breathepage():
    # Redirect to the 'page2breathe' route
    return redirect(url_for('breathepage'))

@app.route('/rd')
def rd():
    prediction_text_placeholder = "Please wait for processor response"
    return render_template('Raw_signalBreathe.HTML',sample_placeholder=prediction_text_placeholder)
@app.route('/redirect_to_rd')
def redirect_to_rd():
    return redirect(url_for('rd'))

@app.route('/fvc')
def fvc():
    prediction_text_placeholder = "Please wait for processor response"
    return render_template('FVC.HTML',sample_placeholder=prediction_text_placeholder)
@app.route('/redirect_to_fvc')
def redirect_to_fvc():
    return redirect(url_for('fvc'))

@app.route('/svc')
def svc():
    prediction_text_placeholder = "Please wait for processor response"
    return render_template('SVC.HTML',sample_placeholder=prediction_text_placeholder)
@app.route('/redirect_to_svc')
def redirect_to_svc():
    return redirect(url_for('svc'))

@app.route('/mvv')
def mvv():
    prediction_text_placeholder = "Please wait for processor response"
    return render_template('MVV.HTML',sample_placeholder=prediction_text_placeholder)
@app.route('/redirect_to_mvv')
def redirect_to_mvv():
    return redirect(url_for('mvv'))

@app.route('/sm')
def sm():
    prediction_text_placeholder = "Please wait for processor response"
    return render_template('Standard_Metrics_Breathe.HTML',sample_placeholder=prediction_text_placeholder)
@app.route('/sm')
def redirect_to_sm():
    return redirect(url_for('sm'))

@app.route('/doghome')
def doghome():
    prediction_text_placeholder = "Please wait for processor response"
    return render_template('dogBreatheHome.HTML',sample_placeholder=prediction_text_placeholder)
@app.route('/doghome')
def redirect_to_doghome():
    return redirect(url_for('doghome'))

standard_to = StandardScaler()

@app.route("/datamaker", methods=['POST'])
def datamaker():
    if request.method == 'POST':
        folder_path = request.form.get('folder')
        #folder_path = folder_path.split('/')
        folder_path = 'Scripts/Bacteria_Data/'+folder_path
        result = dp1.create_and_move_csv(folder_path)
        #result == "Successfull"
        if result == "Successfull":
            sample_placeholder2 = "Operation Successsfull and masterData data is at"+folder_path+r"\NewFolder\masterData\Masterdata.csv"
            return render_template('DMwebpage.html',sample_placeholder=sample_placeholder2)
        else:
            sample_placeholder2 = "Please enter a valid location"
            return render_template('DMwebpage.html',sample_placeholder=sample_placeholder2)
    else:
        return render_template('page2breathe.html')
@app.route("/dataclustering", methods=['POST'])
def dataclustering():
    if request.method== 'POST':
        visit = request.form.get("visit")
        print(visit)
        if visit == "one":
            folder_path = request.files['csv_file']
            csv_filename = os.path.join(app.config['UPLOAD_FOLDER'], folder_path.filename)
            # Save the uploaded file
            folder_path.save(csv_filename)
            # Read the CSV data into a DataFrame
            dfX = pd.read_csv(csv_filename)
            # Extract unique values from the 'bacteria' column
            unq_bacts = dfX['bacteria'].unique()
            unq_concs =dfX['concentration'].unique()
            unq_vols =dfX['volume'].unique()
            unq_sli =dfX['slide'].unique()
            # Store the file path and DataFrame in the session
            session['csv_file_path'] = csv_filename
            session['clusteringOption'] = request.form.get('clusteringOption')
            strings = unq_bacts.tolist()
            unq_concs = unq_concs.tolist()
            unq_vols=unq_vols.tolist()
            unq_sli = unq_sli.tolist()
            print(unq_concs)
            print(unq_bacts)
            return render_template('clusterwebpage.html',item=strings,item2=unq_concs,item3=unq_vols,item4=unq_sli)
        else:
            print(visit)
            bacts = request.form['selected_values']
            conc= request.form['conc_typeX']
            vol = request.form['vol_typeX']
            slide = request.form['slide_typeX']
            csv_file_path = session.get('csv_file_path')
            clusteringOption = session.get('clusteringOption')
            if csv_file_path:
                # Parse CSV string into DataFrame
                df = pd.read_csv(csv_file_path)
                print(df.head())
                result = clustering.Clustered_data(df,clusteringOption,bacts,conc,vol,slide)
                sample_placeholder2 = "clustering Successfull"
                return render_template('clusterwebpage.html',sample_placeholder=sample_placeholder2)
            else:
                plot="No data Availiable"
                return render_template('clusterwebpage.html',plot=plot)
    else:
        return render_template('page2breathe.html')
@app.route("/datavisualization",methods=['post'])
def datavisualization():
    if request.method=='POST':
        visit = request.form.get("visit")
        if visit == "one":
            folder_path = request.files['csv_file']
            data_type=request.form.get("data_type")
            if data_type=='cluster_data':
                 csv_filename = os.path.join(app.config['UPLOAD_FOLDER'], folder_path.filename)
                 folder_path.save(csv_filename)
                 dfX = pd.read_csv(csv_filename)
                 unq_bacts = dfX['bacteria'].unique()
                 strings = unq_bacts.tolist()
                 session['csv_file_path'] = csv_filename
                 session['data_type'] = data_type
                 return render_template('visual.html',item=strings)
            print("raw data")
            csv_filename = os.path.join(app.config['UPLOAD_FOLDER'], folder_path.filename)
            folder_path.save(csv_filename)
            dfX = pd.read_csv(csv_filename)
            unq_bacts = dfX['bacteria'].unique()
            unq_concs =dfX['concentration'].unique()
            unq_vols =dfX['volume'].unique()
            #unq_sli =dfX['slide'].unique()
            unq_tri =dfX['trail'].unique()

            session['csv_file_path'] = csv_filename
            
            session['data_type'] = data_type
            strings = unq_bacts.tolist()
            unq_concs = unq_concs.tolist()
            unq_vols=unq_vols.tolist()
            #unq_sli = unq_sli.tolist()
            unq_tri=unq_tri.tolist()
            print(unq_concs)
            print(unq_bacts)
            return render_template('visual.html',item=strings,item2=unq_concs,item3=unq_vols,item4=["abc"],item5=unq_tri)
        else:
            print("entered else")
            print(request.form)
            csv_file_path = session.get('csv_file_path')
            d_type = session.get('data_type')
            if d_type=='cluster_data':
                if csv_file_path:
                    df = pd.read_csv(csv_file_path)
                    print(df.head())
                    bacts=request.form.getlist('options[]')
                    img_base64 = VSUL.clustered_plotter(df,bacts)
                    return render_template('visual.html',img_base64=img_base64)
                else:
                    plot="No data Availiable"
                    return render_template('visual.html',plot=plot)
            print("raw data")
            bacts=request.form.getlist('options[]')
            conc= request.form['conc_typeX']
            vol = request.form['vol_typeX']
            slide = request.form['slide_typeX']
            trails = request.form.getlist('options2[]')

            
            plot_type = request.form['plot_type']
            print(request.form['conc_typeX'])
            print(request.form['vol_typeX'])
            print(request.form['slide_typeX'])
            print(request.form.getlist('options[]'))
            print(request.form.getlist('options2[]'))
           
            #print(request.form['cb_trails'])
            if csv_file_path:
                # Parse CSV string into DataFrame
                df = pd.read_csv(csv_file_path)
                print(df.head())
                img_base64 = VSUL.visual(df,bacts,d_type,plot_type,conc,vol,slide,trails)
                return render_template('visual.html',img_base64=img_base64)
            else:
                plot="No data Availiable"
                return render_template('visual.html',plot=plot)
    else:
        return render_template('page2breathe.html')

@app.route('/dataML',methods=['post'])
def dataML():
    if request.method=='POST':
        print("reached ML fun")
        print(request.form)
        ML_type = request.form['radioOption']
        if ML_type=="ML_train":
            visit = request.form.get("visit")
            if visit == "one":
                folder_path = request.files['csv_file']
                data_type = request.form.get("Train_ML")
                csv_filename = os.path.join(app.config['UPLOAD_FOLDER'], folder_path.filename)
                folder_path.save(csv_filename)
                dfX = pd.read_csv(csv_filename)
                unq_bacts = dfX['bacteria'].unique()
                unq_bacts=unq_bacts.tolist()
                print(unq_bacts)
                session['csv_file_path'] = csv_filename 
                session['data_type'] = data_type
                #print(data_type)
                #img_base64 = ML.Training_ML(csv_file,data_type)
                sample_placeholder2 = "ML Operation Successfull"
                return render_template('ML_webpage.html',item=unq_bacts)
            else:
                csv_file_path = session.get('csv_file_path')
                d_type = session.get('data_type')
                if csv_file_path:
                    df = pd.read_csv(csv_file_path)
                    print(df.head())
                    bacts=request.form.getlist('options')
                    img_base64 = ML.Training_ML(df,d_type,bacts)
                    return render_template('ML_webpage.html',img_base64=img_base64)
                else:
                    plot="No data Availiable"
                    return render_template('ML_webpage.html',plot=plot)
        else:
            print("entered_testing")
            data = request.files['csv_file']
            data=pd.read_csv(data)
            res=ML.testing(data)
            return render_template('ML_webpage.html',img_base64=img_base64)
    else:
        return render_template('page2breathe.html')
    
@app.route('/rawData',methods=['post'])
def rawData():
    if request.method == 'POST':
        csv_file = request.files['csv_file']
        df = pd.read_csv(csv_file)
        result = raw_data_calc(df)
        return render_template('Raw_signalBreathe.html',img_base64=result)
    else:
        return render_template('BreatheAnalysisHome.html')

@app.route('/spiro',methods=['post'])
def spiro():
    referrer = request.referrer
    if request.method == 'POST':
        csv_file = request.files['csv_file']
        analysis_type = request.form['radioOption']
        if analysis_type=="fvc":
            print("FVC")
            df = pd.read_csv(csv_file)
            plot1, plot2=fvc_calc(df)
            return render_template('FVC.html',img_base64_1=plot1,img_base64_2=plot2)
        elif analysis_type=="svc" :
            print("SVC")
            df = pd.read_csv(csv_file)
            result=svc_calc(df)
            return render_template('SVC.html',img_base64=result)
        elif analysis_type=="mvv" :
            print("MVV")
            df = pd.read_csv(csv_file)
            result=mvv_calc(df)
            return render_template('MVV.html',img_base64=result)
        elif analysis_type=="sm" :
            print("SM")
            df = pd.read_csv(csv_file)
            resp_rate,LungCapacity, Quality=sm_calc(df)
            return render_template('Standard_Metrics_Breathe.html',resp_rate=resp_rate,LungCapacity=LungCapacity,Quality=Quality)
    else:
        print(referrer)
        print("Not post")
        return redirect(referrer or url_for('breathepage'))
        #return render_template('BreatheAnalysisHome.html')


                    ################################################################################
                    ##############################     NEW ANALYSIS     ###########################
                    ################################################################################


@app.route("/NewVisualization",methods=['post'])
def NewVisualization():
    if request.method=='POST':
        visit = request.form.get("visit")
        if visit == "one":
            folder_path = request.files['csv_file']
            data_type=request.form.get("data_type")
            csv_filename = os.path.join(app.config['UPLOAD_FOLDER'], folder_path.filename)
            folder_path.save(csv_filename)
            dfX = pd.read_csv(csv_filename)
            unq_bacts = dfX['bacteria'].unique()
            unq_concs =dfX['concentration'].unique()
            unq_vols =dfX['volume'].unique()
            unq_sli =dfX['slide'].unique()
            unq_tri =dfX['trial'].unique()

            session['csv_file_path'] = csv_filename
            
            session['data_type'] = data_type
            strings = unq_bacts.tolist()
            unq_concs = unq_concs.tolist()
            unq_vols=unq_vols.tolist()
            unq_sli = unq_sli.tolist()
            unq_tri=unq_tri.tolist()
            print(unq_concs)
            print(unq_bacts)
            return render_template('current_analysis_rawDatavisual.HTML',item=strings,item2=unq_concs,item3=unq_vols,item4=unq_sli,item5=unq_tri)
        else:
            print("entered else")
            print(request.form)
            csv_file_path = session.get('csv_file_path')
            d_type = session.get('data_type')
            print("raw data")
            bacts=request.form.getlist('options[]')
            conc= request.form['conc_typeX']
            vol = request.form.getlist('vol_typeX[]')
            slide = request.form.getlist('options3[]')
            trails = request.form.getlist('options2[]')

            
            #plot_type = request.form['plot_type']
            print(request.form['conc_typeX'])
            print(request.form.getlist('vol_typeX[]'))
           # print(request.form['slide_typeX'])
            print(request.form.getlist('options[]'))
            print(request.form.getlist('options2[]'))
           
            #print(request.form['cb_trails'])
            if csv_file_path:
                # Parse CSV string into DataFrame
                df = pd.read_csv(csv_file_path)
                print(df.head())    
                img_base64 = main_function_visual(df,bacts,conc,vol,slide,trails)
                return render_template('current_analysis_rawDatavisual.html',img_base64=img_base64)
            else:
                plot="No data Availiable"
                return render_template('current_analysis_rawDatavisual.html',plot=plot)
    else:
        return render_template('new_analysis_landing.html')

@app.route("/pcaMethod",methods=['post'])
def pcaMethod():
    if request.method=='POST':
        visit = request.form.get("visit")
        if visit == "one":
            folder_path = request.files['csv_file']
            data_type=request.form.get("data_type")
            csv_filename = os.path.join(app.config['UPLOAD_FOLDER'], folder_path.filename)
            folder_path.save(csv_filename)
            dfX = pd.read_csv(csv_filename)
            unq_bacts = dfX['bacteria'].unique()
            unq_concs =dfX['concentration'].unique()
            unq_vols =dfX['volume'].unique()
            unq_sli =dfX['slide'].unique()
            unq_tri =dfX['trial'].unique()

            session['csv_file_path'] = csv_filename
            
            session['data_type'] = data_type
            strings = unq_bacts.tolist()
            unq_concs = unq_concs.tolist()
            unq_vols=unq_vols.tolist()
            unq_sli = unq_sli.tolist()
            unq_tri=unq_tri.tolist()
            print(unq_concs)
            print(unq_bacts)
            return render_template('current_analysis_PCA.HTML',item=strings,item2=unq_concs,item3=unq_vols,item4=unq_sli,item5=unq_tri)
        else:
            print("entered else")
            print(request.form)
            csv_file_path = session.get('csv_file_path')
            d_type = session.get('data_type')
            print("raw data")
            bacts=request.form.getlist('options[]')
            conc= request.form['conc_typeX']
            vol = request.form['vol_typeX']
            slide = request.form.getlist('options3[]')
            trails = request.form.getlist('options2[]')

            
            #plot_type = request.form['plot_type']
            print(request.form['conc_typeX'])
            print(request.form['vol_typeX'])
           # print(request.form['slide_typeX'])
            print(request.form.getlist('options[]'))
            print(request.form.getlist('options2[]'))
           
            #print(request.form['cb_trails'])
            if csv_file_path:
                # Parse CSV string into DataFrame
                df = pd.read_csv(csv_file_path)
                print(df.head())    
                img_base64 = main_function_pca(df,bacts,conc,vol,slide,trails)
                return render_template('current_analysis_PCA.html',img_base64=img_base64)
            else:
                plot="No data Availiable"
                return render_template('current_analysis_PCA.html',plot=plot)
    else:
        return render_template('new_analysis_landing.html')

@app.route("/MLMethod",methods=['post'])
def MLMethod():
    if request.method=='POST':
        visit = request.form.get("visit")
        if visit == "one":
            folder_path = request.files['csv_file']
            data_type=request.form.get("data_type")
            csv_filename = os.path.join(app.config['UPLOAD_FOLDER'], folder_path.filename)
            folder_path.save(csv_filename)
            dfX = pd.read_csv(csv_filename)
            unq_bacts = dfX['bacteria'].unique()
            unq_concs =dfX['concentration'].unique()
            unq_vols =dfX['volume'].unique()
            unq_sli =dfX['slide'].unique()
            unq_tri =dfX['trial'].unique()

            session['csv_file_path'] = csv_filename
            
            session['data_type'] = data_type
            strings = unq_bacts.tolist()
            unq_concs = unq_concs.tolist()
            unq_vols=unq_vols.tolist()
            unq_sli = unq_sli.tolist()
            unq_tri=unq_tri.tolist()
            print(unq_concs)
            print(unq_bacts)
            return render_template('current_analysis_ML.HTML',item=strings,item2=unq_concs,item3=unq_vols,item4=unq_sli,item5=unq_tri)
        else:
            print("entered else")
            print(request.form)
            csv_file_path = session.get('csv_file_path')
            d_type = session.get('data_type')
            print("raw data")
            bacts=request.form.getlist('options[]')
            conc= request.form['conc_typeX']
            vol = request.form['vol_typeX']
            slide = request.form.getlist('options3[]')
            trails = request.form.getlist('options2[]')

            
            #plot_type = request.form['plot_type']
            print(request.form['conc_typeX'])
            print(request.form['vol_typeX'])
           # print(request.form['slide_typeX'])
            print(request.form.getlist('options[]'))
            print(request.form.getlist('options2[]'))
           
            #print(request.form['cb_trails'])
            if csv_file_path:
                # Parse CSV string into DataFrame
                df = pd.read_csv(csv_file_path)
                print(df.head())    
                img_base64 = plotter_svm(df,bacts,conc,vol,slide,trails)
                return render_template('current_analysis_ML.html',img_base64=img_base64)
            else:
                plot="No data Availiable"
                return render_template('current_analysis_ML.html',plot=plot)
    else:
        return render_template('new_analysis_landing.html')

if __name__=="__main__":
    app.run(debug=False)
    