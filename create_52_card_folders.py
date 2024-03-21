import os 
def main():
    value_list = [str(x+2) for x in range(9)]
    value_list.extend(['K', 'Q', 'J', 'A'])
    suit_list = ['d', 's', 'h', 'c']
    kards_52 = [x + y for x in value_list for y in suit_list
    for card in kards_52:
        os.mkdir(f"52kards/{card}")
                
if __name__ == '__main__':
    main()