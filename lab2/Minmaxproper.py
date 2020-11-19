'''
Kółko i krzyżyk https://pl.wikipedia.org/wiki/K%C3%B3%C5%82ko_i_krzy%C5%BCyk
Minmax Jerzy Rześniowiecki, Szymon Maj 
'''
from math import inf as infinity
from random import choice
import platform
import time
from os import system

HUMAN = -1
COMP = +1
board = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
]


def evaluate(state):
    """
    Function checks based on the board state if there's a winner and who is it
    Parameters: 
        state (list): Current state of the game board
    Returns:
        int: +1 for AI victory, -1 for player/Human victory, 0 for draw
    """
    if wins(state, COMP):
        score = +1
    elif wins(state, HUMAN):
        score = -1
    else:
        score = 0

    return score

def wins(state, player):
    """
    Function checks if player managed to occupy 3 squares in straight line.
        
    Parameters: 
        state (list): Current state of the game board
        player (int): id of the player "+1" for comp "-1" for human
    
    Returns:
        bool: Informs if the player won the game
    """
    win_state = [
        [state[0][0], state[0][1], state[0][2]],
        [state[1][0], state[1][1], state[1][2]],
        [state[2][0], state[2][1], state[2][2]],
        [state[0][0], state[1][0], state[2][0]],
        [state[0][1], state[1][1], state[2][1]],
        [state[0][2], state[1][2], state[2][2]],
        [state[0][0], state[1][1], state[2][2]],
        [state[2][0], state[1][1], state[0][2]],
    ]
    if [player, player, player] in win_state:
        return True
    else:
        return False

def game_over(state):
    """
    Function calls wins() function for both players to check if the game has ended
    
    Parameters: 
        state (list): Current state of the game board
        
    Returns:
        bool: Informs if either of players won the game
    """
    return wins(state, HUMAN) or wins(state, COMP)

def empty_cells(state):
    """
    Function marks empty cells for valid move options
    
    Parameters: 
        state (list): Current state of the game board
        
    Returns:
        array: Lists all valid moves
    """
    cells = []

    for x, row in enumerate(state):
        for y, cell in enumerate(row):
            if cell == 0:
                cells.append([x, y])

    return cells

def set_move(x, y, player):
    """
    Attempts to execute players moves depending on the valid move list from empty_cells(board)
    
    Parameters: 
        x,y(int):vertical and horizontal positions of the squares
        player (int): id of the player "+1" for comp "-1" for human
        
    Returns:
        bool: Either declines the move if it is not in a valid position, or returns it as valid and sets the mark down on the square selected
    """
    if [x, y] in empty_cells(board):
        board[x][y] = player
        return True
    else:
        return False

def minimax(state, depth, player):
    """
    Function running the AI calculations upon the game dependant on it's depth, current player, and game state.
    
    Parameters: 
        state (list): Current state of the game board
        depth (int): Current stage of the game
        player (int): id of the player "+1" for comp "-1" for human
        
    Returns:
        array: Most optimal move in accordance to the minimax algorithm
    """
    if player == COMP:
        best = [-1, -1, -infinity]
    else:
        best = [-1, -1, +infinity]

    if depth == 0 or game_over(state):
        score = evaluate(state)
        return [-1, -1, score]

    for cell in empty_cells(state):
        x, y = cell[0], cell[1]
        state[x][y] = player
        score = minimax(state, depth - 1, -player)
        state[x][y] = 0
        score[0], score[1] = x, y

        if player == COMP:
            if score[2] > best[2]:
                best = score  # max value
        else:
            if score[2] < best[2]:
                best = score  # min value

    return best

def clean():
    """
    Function calls system funtion to refresh the screen
    
    Parameters: 
        null
        
    Returns:
        null
    """
    os_name = platform.system().lower()
    if 'windows' in os_name:
        system('cls')
    else:
        system('clear')

def render(state, c_choice, h_choice):
    """
    Function draws game state in the console window
    
    Parameters: 
        state (list): Current state of the game board
        c_choice (string): Character used to represent areas occupied by computer
        h_choice (string): Character used to represent areas occupied by human
        
    Returns:
        null
    """
    chars = {
        -1: h_choice,
        +1: c_choice,
        0: ' '
    }
    str_line = '---------------'
    print('\n' + str_line)
    for row in state:
        for cell in row:
            symbol = chars[cell]
            print(f'| {symbol} |', end='')
        print('\n' + str_line)

def ai_turn(c_choice, h_choice):
    """
    Algorithm for Ai to decide it's next move. Calls set_move() function to set it's move.
    
    Parameters: 
        c_choice (string): Character used to represent areas occupied by computer
        h_choice (string): Character used to represent areas occupied by human
    Returns:
        function: set_move()
    """
    depth = len(empty_cells(board))
    if depth == 0 or game_over(board):
        return

    clean()
    print(f'Tura AI [{c_choice}]')
    render(board, c_choice, h_choice)

    if depth == 9:
        x = choice([0, 1, 2])
        y = choice([0, 1, 2])
    else:
        move = minimax(board, depth, COMP)
        x, y = move[0], move[1]

    set_move(x, y, COMP)
    time.sleep(1)

def human_turn(c_choice, h_choice):
    """
    Function prompts human player for his next move. Calls set_move() function to set player's move.
    
    Parameters: 
        c_choice (string): Character used to represent areas occupied by computer
        h_choice (string): Character used to represent areas occupied by human
    Returns:
        function: set_move()
    """    
    depth = len(empty_cells(board))
    if depth == 0 or game_over(board):
        return

    # Dictionary of valid moves
    move = -1
    moves = {
        1: [0, 0], 2: [0, 1], 3: [0, 2],
        4: [1, 0], 5: [1, 1], 6: [1, 2],
        7: [2, 0], 8: [2, 1], 9: [2, 2],
    }

    clean()
    print(f'Tura Gracza [{h_choice}]')
    render(board, c_choice, h_choice)

    while move < 1 or move > 9:
        try:
            move = int(input('Use numpad (1..9): '))
            coord = moves[move]
            can_move = set_move(coord[0], coord[1], HUMAN)

            if not can_move:
                print('Bledne pole')
                move = -1
        except (EOFError, KeyboardInterrupt):
            print('Error')
            exit()
        except (KeyError, ValueError):
            print('Wybierz z zakresu')

def main():
    clean()
    h_choice = ''  # X or O
    c_choice = ''  # X or O
    first = ''  # if human is the first

    # Human chooses X or O to play
    while h_choice != 'O' and h_choice != 'X':
        try:
            print('')
            h_choice = input('Wybierz X albo O\nChosen: ').upper()
        except (EOFError, KeyboardInterrupt):
            print('Error')
            exit()
        except (KeyError, ValueError):
            print('Wybierz z zakresu')

    # Setting computer's choice
    if h_choice == 'X':
        c_choice = 'O'
    else:
        c_choice = 'X'

    # Human may starts first
    clean()
    while first != 'Y' and first != 'N':
        try:
            first = input('Czy chcesz byc pierwszy?[y/n]: ').upper()
        except (EOFError, KeyboardInterrupt):
            print('Error')
            exit()
        except (KeyError, ValueError):
            print('Wybierz z zakresu')

    # Main loop of this game
    while len(empty_cells(board)) > 0 and not game_over(board):
        if first == 'N':
            ai_turn(c_choice, h_choice)
            first = ''

        human_turn(c_choice, h_choice)
        ai_turn(c_choice, h_choice)

    # Game over message
    if wins(board, HUMAN):
        clean()
        print(f'Tura Gracza [{h_choice}]')
        render(board, c_choice, h_choice)
        print('Wygrales!')
    elif wins(board, COMP):
        clean()
        print(f'Tura AI [{c_choice}]')
        render(board, c_choice, h_choice)
        print('Przegrales!')
    else:
        clean()
        render(board, c_choice, h_choice)
        print('Remis!')


while 1:
    main()
    if input("Jeszcze raz? y/n: ").lower() == "n":
        break
    else:
        board = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
            ]
        