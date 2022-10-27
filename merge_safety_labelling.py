import pickle
import pandas as pd, os
path = './Samples'
mechs = ['EC1', 'CE', 'ECE', 'DISP', 'ECP', 'T', 'E'] 
for mech in mechs: 
   files = os.listdir(f'{path}/{mech}')
   files = [f for f in files if f.endswith('.txt')]
   merge_safe = {}
   conditional_safe = {}
   for file in files:
      df = pd.read_csv(f'{path}/{mech}/{file}')
      df = df.loc[df['v'] == min(df['v'])]
      n = len(df)//2
      fwd = df.iloc[:n, :]
      rev = df.iloc[n:, :]
      d_fwd = abs(fwd.loc[fwd.index[-100], 'A'] - fwd.loc[fwd.index[-1], 'A'])
      d_rev = abs(rev.loc[rev.index[-100], 'A'] - rev.loc[rev.index[-1], 'A'])
      left_merge = d_rev <= 0.05
      right_merge = d_fwd <= 0.05
      if left_merge and right_merge:
         merge_safe[file] = (left_merge, right_merge)
      elif left_merge or right_merge:
         conditional_safe[file] = (left_merge, right_merge)
   pickle.dump(merge_safe, open(f'./Merge_Safety/{mech}_safe.pkl','wb'))
   pickle.dump(conditional_safe, open(f'./Merge_Safety/{mech}_half_safe.pkl','wb'))
   print(mech)
   print(f'Safe: {len(merge_safe)}')
   print(f'Half Safe: {len(conditional_safe)}')
   print(f'Neither: {1-((len(merge_safe)+len(conditional_safe))/len(files))}')