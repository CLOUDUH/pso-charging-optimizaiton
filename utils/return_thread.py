'''
Author: CLOUDUH
Date: 2022-06-26 13:52:54
LastEditors: CLOUDUH
LastEditTime: 2022-06-26 16:35:33
Description: 
'''

import threading, time
from threading import Thread

class ReturnThread(Thread):
    '''With ReturnThread, we can get the result of the function
    Args:
        func: function
        args: arguments
    Functions:
        run: run the function
        get_result: get the result of the function
    '''

    def __init__(self, func, args=()):
        super(ReturnThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        '''Run the function'''
        self.result = self.func(*self.args)

    def get_result(self):
        '''Get the result of the function'''
        Thread.join(self)  
        try:
            return self.result
        except Exception:
            return None

def admin(number):
    uiu = number
    for i in range(10):
        uiu = uiu+i
    return uiu
 
if __name__ == "__main__":
    
    for i in range(10):

        threads = []

        for j in range(4):
            exec(f'more_th{j} = ReturnThread(admin,({j*j},))')
            exec(f'threads.append(more_th{j})')

        for j in range(4): 
            threads[j].start() # Parallel computing

        for j in range(4):
            threads[j].join() # Wait all thread to finish
        
        for j in range(4):
            print(threads[j].get_result())
            
        print(threads)
