import pandas as pd

# CSV 파일 읽어오기
df = pd.read_csv('C:/Users/ATMOS/Desktop/random_test/random_result.csv')

# A_111123 열의 행 값이 Raw_111123 열의 행 값과 같은 행 삭제
df.loc[df['A_111123'] == df['Raw_111123'], 'A_111123'] = ''
df.loc[df['B_111123'] == df['Raw_111123'], 'B_111123'] = ''
df.loc[df['C_111123'] == df['Raw_111123'], 'C_111123'] = ''
df.loc[df['D_111123'] == df['Raw_111123'], 'D_111123'] = ''
df.loc[df['E_111123'] == df['Raw_111123'], 'E_111123'] = ''
df.loc[df['F_111123'] == df['Raw_111123'], 'F_111123'] = ''
df.loc[df['G_111123'] == df['Raw_111123'], 'G_111123'] = ''
df.loc[df['H_111123'] == df['Raw_111123'], 'H_111123'] = ''

df.loc[df['A_131222'] == df['Raw_131222'], 'A_131222'] = ''
df.loc[df['B_131222'] == df['Raw_131222'], 'B_131222'] = ''
df.loc[df['C_131222'] == df['Raw_131222'], 'C_131222'] = ''
df.loc[df['D_131222'] == df['Raw_131222'], 'D_131222'] = ''
df.loc[df['E_131222'] == df['Raw_131222'], 'E_131222'] = ''
df.loc[df['F_131222'] == df['Raw_131222'], 'F_131222'] = ''
df.loc[df['G_131222'] == df['Raw_131222'], 'G_131222'] = ''
df.loc[df['H_131222'] == df['Raw_131222'], 'H_131222'] = ''


# 수정된 DataFrame을 CSV 파일로 저장
df.to_csv('C:/Users/ATMOS/Desktop/random_test/random_result2.csv', index=False)
