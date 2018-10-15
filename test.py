# import requests
# a = ",".join(str(item) for item in range(50))
# b = a.split(",")
# print(b)
# res = requests.post('http://localhost:5000/api/predict', json={"input":a,"model":"BKNET"})
# if res.ok:
#     print (res.json())

'''
SA: 38, 47
BKNET: 5, 15, 36 
'''
# b = str(print('a'))
a = "print('aaa')"
exec(a)