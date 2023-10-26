from django.http import HttpResponse
from django.shortcuts import render
import joblib
import pandas as pd

def home(request):
    return render(request,"home.html")

def result(request):
    
    k_fit=joblib.load("finalized_model.sav")
    
    list1=[]
    
    list1.append(request.GET['EXT1'])
    list1.append(request.GET['EXT2'])
    list1.append(request.GET['EXT3'])
    list1.append(request.GET['EXT4'])
    list1.append(request.GET['EXT5'])
    list1.append(request.GET['EXT6'])
    list1.append(request.GET['EXT7'])
    list1.append(request.GET['EXT8'])
    list1.append(request.GET['EXT9'])
    list1.append(request.GET['EXT10'])
    list1.append(request.GET['EST1'])
    list1.append(request.GET['EST2'])
    list1.append(request.GET['EST3'])
    list1.append(request.GET['EST4'])
    list1.append(request.GET['EST5'])
    list1.append(request.GET['EST6'])
    list1.append(request.GET['EST7'])
    list1.append(request.GET['EST8'])
    list1.append(request.GET['EST9'])
    list1.append(request.GET['EST10'])
    list1.append(request.GET['AGR1'])
    list1.append(request.GET['AGR2'])
    list1.append(request.GET['AGR3'])
    list1.append(request.GET['AGR4'])
    list1.append(request.GET['AGR5'])
    list1.append(request.GET['AGR6'])
    list1.append(request.GET['AGR7'])
    list1.append(request.GET['AGR8'])
    list1.append(request.GET['AGR9'])
    list1.append(request.GET['AGR10'])
    list1.append(request.GET['CSN1'])
    list1.append(request.GET['CSN2'])
    list1.append(request.GET['CSN3'])
    list1.append(request.GET['CSN4'])
    list1.append(request.GET['CSN5'])
    list1.append(request.GET['CSN6'])
    list1.append(request.GET['CSN7'])
    list1.append(request.GET['CSN8'])
    list1.append(request.GET['CSN9'])
    list1.append(request.GET['CSN10'])
    list1.append(request.GET['OPN1'])
    list1.append(request.GET['OPN2'])
    list1.append(request.GET['OPN3'])
    list1.append(request.GET['OPN4'])
    list1.append(request.GET['OPN5'])
    list1.append(request.GET['OPN6'])
    list1.append(request.GET['OPN7'])
    list1.append(request.GET['OPN8'])
    list1.append(request.GET['OPN9'])
    list1.append(request.GET['OPN10'])

    print(list1)
    
    ans=k_fit.predict([list1])
    
    int_list = [int(str_num) for str_num in list1]
    print(int_list)
    
    ext = sum(int_list[0:10])/10
    est = sum(int_list[10:20])/10
    agr = sum(int_list[20:30])/10
    csn = sum(int_list[30:40])/10
    opn = sum(int_list[40:50])/10
    
    if ans==[0]:
        traits="Extrovert and Conscientious person"
    elif ans==[1]:
        traits="Openness person with low agreeableness and conscientiousness"
    elif ans==[2]:
        traits="Extrovert and Conscientious with low Agreeableness"
    elif ans==[3]:
        traits="Extrovert,Neurotic and Open to Experience"
    else:
        traits="Extrovert and Conscientious"
 
    return render(request,"result.html",{'ans':ans,'int_list':int_list,'ext':ext,'est':est,'agr':agr,'csn':csn,'opn':opn,'traits':traits})