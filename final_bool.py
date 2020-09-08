'''
# sort this in inc order of topleft x of bb
# not gate processing
'''

import cv2 as cv2
import os
import numpy as np



def find_boolean_exp(bb_x,predictions):
	#value of initial input wires
	inp="a"
	for i in range(len(predictions)):
		flag=0
		if(predictions[i]['label']=="wire"):		
			x=predictions[i]['topleft']
			for j in range(len(predictions)):
				if(predictions[j]['bottomright']['x']==x['x']):
					flag=1				
	
			if(flag==0):
				predictions[i]['value']=inp
				inp=chr(ord(inp)+1)
	
	for i in range(len(predictions)):
		btm_r=predictions[i]['bottomright']
		if('value' in predictions[i]):		
			val=predictions[i]['value']
			for j in range(len(predictions)):
				top_l=predictions[j]['topleft']	
							
				if((top_l['x']==btm_r['x']) and (top_l['y']==btm_r['y'])):
					print('+++++++++++++++++++++++++++')
					predictions[j]['value']=val		
		
	#print(predictions)
	
	for i in range(len(predictions)):
		x1=predictions[i]['topleft']['x']
		y1=predictions[i]['topleft']['y']
		
		x2=predictions[i]['bottomright']['x']
		y2=predictions[i]['bottomright']['y']
#		if('value' in predictions[i]):
#			print('label',predictions[i]['label'],(x1,y1),'-----',(x2,y2),'value=',predictions[i]['value'])
#		else:
#			print('label',predictions[i]['label'],(x1,y1),'-----',(x2,y2))	
								
	i=0								
	#processing gates from left to right
	print('NOW CHECK')
	#print(predictions)
	for i in range(len(predictions)):
		inp_wires_val=[]
		if(predictions[i]['label']=="not"):
			continue
		if(predictions[i]['label']!="wire"):
			gate=predictions[i]['label']
			print('GATE IS :',gate)
			left_x_bb=predictions[i]['topleft']['x']
			right_x_bb=predictions[i]['bottomright']['x']
			top_y_bb=predictions[i]['topleft']['y']-5
			bottom_y_bb=predictions[i]['bottomright']['y']+5
			#print('top x',left_x_bb)
			#get val of inp wires			
			for j in range(len(predictions)):
				if((predictions[j]['label']=="wire") and (abs(predictions[j]['bottomright']['x']-left_x_bb)<=5) and (predictions[j]['bottomright']['y'] > top_y_bb) and (predictions[j]['bottomright']['y'] < bottom_y_bb)):
					if('value' in predictions[j]):
						print('+++',predictions[j])
									
						s=predictions[j]['value']
#						print(s)
						inp_wires_val.insert(len(inp_wires_val),predictions[j]['value'])	
					
			print('------------------')
			#print(inp_wires_val)
			
			#get output wire
			out=""
			str=""
			cmpl_out=""
			
			#remove duplicates			
			#inp_wires_val=list(dict.fromkeys(inp_wires_val))
			final_list = [] 
			if(predictions[i+1]['label']=="not"):
				print('VVVVVVVVVVVVVV')				
				left_x_bb1=predictions[i+1]['topleft']['x']
				right_x_bb1=predictions[i+1]['bottomright']['x']
				top_y_bb1=predictions[i+1]['topleft']['y']-5
				bottom_y_bb1=predictions[i+1]['bottomright']['y']+5
				
				if(gate=="and"):
					print('NANDDDD')
					for y in range(len(inp_wires_val)):
						str+=inp_wires_val[y]+"."
						l=len(str)-1
						out=str[0:l]
						predictions[i]['output']=out
						cmpl_out='('+out+')'+'\''
						print(cmpl_out)
					for r in range(len(predictions)):
						if((predictions[r]['label']=="wire") and (abs(predictions[r]['topleft']['x']-right_x_bb1)<=5) and (predictions[r]['topleft']['y'] > top_y_bb1) and (predictions[r]['topleft']['y'] < bottom_y_bb1)):	
							predictions[r]['value']=cmpl_out
							predictions[i+1]['output']=cmpl_out
							#print(predictions)
					
					for u in range(len(predictions)):
						btm_r=predictions[u]['bottomright']
						if('value' in predictions[u]):
							val=predictions[u]['value']
							for g in range(len(predictions)):
								top_l=predictions[g]['topleft']
								if((top_l['x']==btm_r['x']) and (top_l['y']==btm_r['y'])):
									print('+++++++++++++++++++++++++++')
									predictions[g]['value']=val
											
					#print(predictions)

				
				if(gate=="or"):
					print('NORRRR')
					for y in range(len(inp_wires_val)):
						str+=inp_wires_val[y]+"+"
						l=len(str)-1
						out=str[0:l]
						predictions[i]['output']=out
						cmpl_out='('+out+')'+'\''
						#print(cmpl_out)
					for r in range(len(predictions)):
						if((predictions[r]['label']=="wire") and (abs(predictions[r]['topleft']['x']-right_x_bb1)<=5) and (predictions[r]['topleft']['y'] > top_y_bb1) and (predictions[r]['topleft']['y'] < bottom_y_bb1)):	
							predictions[r]['value']=cmpl_out
							predictions[i+1]['output']=cmpl_out
						#print(predictions)
						
					for u in range(len(predictions)):
						btm_r=predictions[u]['bottomright']
						if('value' in predictions[u]):
							val=predictions[u]['value']
							for g in range(len(predictions)):
								top_l=predictions[g]['topleft']
								if((top_l['x']==btm_r['x']) and (top_l['y']==btm_r['y'])):
									print('+++++++++++++++++++++++++++')
									predictions[g]['value']=val
											
#					print(predictions)
				
				
				#i+=1
			else:
				for k in range(len(predictions)):
					
					if((predictions[k]['label']=="wire") and (abs(predictions[k]['topleft']['x']-right_x_bb)<=5) and (predictions[k]['topleft']['y'] > top_y_bb) and (predictions[k]['topleft']['y'] < bottom_y_bb)):	
						print('444444444444')	
						if(gate=="and"):
							print('ANDDDD')
							for y in range(len(inp_wires_val)):
								str+=inp_wires_val[y]+"."
								l=len(str)-1
								out=str[0:l]
								out='('+out+')'
								#print('tttttttttttttttttttt',predictions[i+1]['label'])
								predictions[k]['value']=out
								predictions[i]['output']=out
							for p in range(len(predictions)):
								
								btm_r=predictions[p]['bottomright']
								if('value' in predictions[p]):
									val=predictions[p]['value']
							
									
									for m in range(len(predictions)):
										top_l=predictions[m]['topleft']
										if((top_l['x']==btm_r['x']) and (top_l['y']==btm_r['y'])):									
											print('+++++++++++++++++++++++++++')
											predictions[m]['value']=val				
																					
							#print(predictions)
							for i in range(len(predictions)):
								x1=predictions[i]['topleft']['x']
								y1=predictions[i]['topleft']['y']
								#print('label',predictions[i]['label'],(x1,y1))	
								
								x2=predictions[i]['bottomright']['x']
								y2=predictions[i]['bottomright']['y']
								#print('label',predictions[i]['label'],(x2,y2))	
								
										
																												
							
								
			
													


						elif(gate=="or"):
							print('ORRRR')
							for y in range(len(inp_wires_val)):
								str+=inp_wires_val[y]+"+"
								l=len(str)-1
								out=str[0:l]
								out='('+out+')'
								print('tttttttttttttttttttt',predictions[i+1]['label'])
								predictions[k]['value']=out
								predictions[i]['output']=out
							for p in range(len(predictions)):
								btm_r=predictions[p]['bottomright']
								if('value' in predictions[p]):
									val=predictions[p]['value']
									for m in range(len(predictions)):
										top_l=predictions[m]['topleft']
										if((top_l['x']==btm_r['x']) and (top_l['y']==btm_r['y'])):
											print('+++++++++++++++++++++++++++')
											predictions[m]['value']=val
#							print(predictions)
			#i+=1	
'''
	for i in range(len(predictions)):
		x1=predictions[i]['topleft']['x']
		y1=predictions[i]['topleft']['y']
		
		x2=predictions[i]['bottomright']['x']
		y2=predictions[i]['bottomright']['y']
		if('value' in predictions[i]):
			print('label',predictions[i]['label'],(x1,y1),'-----',(x2,y2),'value=',predictions[i]['value'])
		elif('output' in predictions[i]):
			print('label',predictions[i]['label'],(x1,y1),'-----',(x2,y2),'output=',predictions[i]['output'])
'''

#predictions=[{'topleft': {'y': 69, 'x': 170}, 'bottomright': {'y': 110, 'x': 211}, 'label': 'and'}, {'topleft': {'y': 82, 'x': 211}, 'bottomright': {'y': 97, 'x': 226}, 'label': 'not'}, {'topleft': {'y': 83, 'x': 54}, 'bottomright': {'y': 83, 'x': 170}, 'label': 'wire'}, {'topleft': {'y': 89, 'x': 226}, 'bottomright': {'y': 89, 'x': 448.0}, 'label': 'wire'}, {'topleft': {'y': 89, 'x': 448.0}, 'bottomright': {'y': 204, 'x': 448.0}, 'label': 'wire'}, {'topleft': {'y': 190, 'x': 504}, 'bottomright': {'y': 231, 'x': 545}, 'label': 'and'}, {'topleft': {'y': 95, 'x': 67}, 'bottomright': {'y': 95, 'x': 170}, 'label': 'wire'}, {'topleft': {'y': 203, 'x': 545}, 'bottomright': {'y': 218, 'x': 560}, 'label': 'not'}, {'topleft': {'y': 204, 'x': 448.0}, 'bottomright': {'y': 204, 'x': 504}, 'label': 'wire'}, {'topleft': {'y': 210, 'x': 560}, 'bottomright': {'y': 210, 'x': 800}, 'label': 'wire'}, {'topleft': {'y': 216, 'x': 74}, 'bottomright': {'y': 216, 'x': 504}, 'label': 'wire'}]



#sort gates from left to right
def gen_expression(pred):
    
    c=0
    res=[]
    print('#########',len(pred))
    for i in range(len(pred)):
     #print(pred[i])   
     if((pred[i]['label']=='wire') and (pred[i]['tlx']!=pred[i]['brx']) and (pred[i]['tly']!=pred[i]['bry'])):
         c=c+1   
         continue
        
     res.append(pred[i])   
    print('..............',c)
    print('#########',len(res))
          
          
    
        
    print(len(res))    
    print('********************************')
    print('EXPRESSION LIST')
	
    predictions=[]
	
    for i in range(len(res)):
        predictions.append({'label':res[i]['label'],'topleft':{'x':res[i]['tlx'], 'y':res[i]['tly']},'bottomright':{'x':res[i]['brx'], 'y':res[i]['bry']}})
		
#	print(predictions)

    for i in range(len(predictions)):
        predictions[i]['topleft']['x']
        y1=predictions[i]['topleft']['y']
		#print('label',components[i]['label'],(x1,y1))	
		
        x2=predictions[i]['bottomright']['x']
        y2=predictions[i]['bottomright']['y']
#		print('label',predictions[i]['label'],(x1,y1),'---',(x2,y2))
	
    gates_list=[]	
    for i in range(len(predictions)): 
        if(predictions[i]['label']!="wire"):
            gates_list.append(predictions[i])
				

    gates_list=sorted(gates_list , key=lambda k: (k['topleft']['x']))
    #print('Gates List')
#	print(gates_list)

    predictions=sorted(predictions, key=lambda k: (k['topleft']['y']))

    for i in range(len(predictions)):
        if((predictions[i]['topleft']['x']==predictions[i]['bottomright']['x']) and (predictions[i]['topleft']['y']==predictions[i]['bottomright']['y'])):
            continue		
        if(predictions[i]['label']=="wire"):
            gates_list.append(predictions[i])
		


    print('Final List')
#	print(gates_list)

    predictions=gates_list
    res1=[]
    rmv_list=[]
    c=0
    for i in range(len(predictions)):
        for j in range(i+1,len(predictions)):
            if((predictions[i]['bottomright']['x']==predictions[j]['bottomright']['x']) and (predictions[i]['bottomright']['y']==predictions[j]['bottomright']['y'])):
                rmv_list.append(i)
                c=c+1
    print('@@@@@@',c) 

    print(rmv_list)
               
    for i in range(len(predictions)):
        if i in rmv_list:
            continue
        res1.append(predictions[i])
    predictions=res1
				
    bb_x=[]
    for i in range(len(predictions)):
        if(str(predictions[i]['label'])!="wire"):
            x=predictions[i]['topleft']['x']
            bb_x.append(x)		
    find_boolean_exp(bb_x,predictions)

    max_x=0
    end_gate_idx=0
    for i in range(len(predictions)):
        if(predictions[i]['label']!="wire"):
            topl_x=predictions[i]['topleft']['x']
            if(topl_x>max_x):
                max_x=topl_x
                end_gate_idx=i

    print(predictions[end_gate_idx])
    print('Boolean Expression of the given circuit :')
    if('output' in predictions[end_gate_idx]):
        print(predictions[end_gate_idx]['output'])
        s=predictions[end_gate_idx]['output']
#		h=len(s)-1
#		s=s[1:h]
        return s
    else:
        print('Gate not detected error!!')
        return "Gate missing!!"

		

