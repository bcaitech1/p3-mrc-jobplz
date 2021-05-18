numbers = [1,2,3,4,5,6,7,8,9,10]

num = int(input('입력 => '))

try :
    print('배열 내부 인덱스 : {}'.format(numbers.index(num)))
except :
    print('{} is not in list'.format(num))