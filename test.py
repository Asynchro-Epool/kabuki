
def test(aa=10):
  pass

try:
   test(bb=10)
except:
    pass

print("bb")
# from joblib import Parallel, delayed

# def get_individual_logp(name, data=1, **kwargs):
#     # 在这里使用kwargs参数
#     # ...
#     arg1 = kwargs.pop('arg1', 99)
#     print(arg1)

# tmp_list = [('name1', 'data1'), ('name2', 'data2')]  # 示例数据
# kwargs = {'arg1': 1, 'arg2': 2}  # 示例关键字参数

# results = Parallel(n_jobs=-1)(
#     delayed(get_individual_logp)(name, data,**kwargs)
#     for name, data in tmp_list
# )

