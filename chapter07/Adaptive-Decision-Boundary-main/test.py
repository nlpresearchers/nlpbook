vars_dict={"data_set":"oos",'known_cls_ratio':"1.0"}
test_results={"Open":1.0,"Known":0.5}
results = dict(test_results,**vars_dict)
print(results)