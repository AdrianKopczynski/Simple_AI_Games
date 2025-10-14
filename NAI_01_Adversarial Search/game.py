from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax

class divisorGame(TwoPlayerGame):
    """
    In turn, the players remove a number which is a divisor of the initial value. The player who gets 1 wins!
    Rules: https://algo.monster/liteproblems/1025

    Enviormental setup: install the newest Python version and easyAI library. Then you can just execute this file.

    Authors: Adrian Kopczyński, Gabriel Francke
    """
    def __init__(self, number, players=None):
        """
        Class Initialization

        Parameters:
        number (int): Initial value of the game
        players (list): List of players object

        Returns:
        None
        """
        self.players = players
        self.number = number
        self.current_player = 1 #Human Player will be first, AI Player is '2'

    def possible_moves(self): 
        """
        Function that is calculating all possible moves for current player

        Returns:
        list:List of possible moves for a player
        """

        return [str(number) for number in range(1,self.number) if self.number%number == 0]
    
    def make_move(self,move):
        """
        Making a move; Subtract move value from the current value

        Parameters:
        move (str): The value that we subtract from the current value

        Returns:
        None
        """
        self.number -= int(move)

    def win(self):
        """
        Check if the current value is equal to 1

        Returns:
        boolean: True if current value is equal to 1 else False
        """
        return self.number==1
    
    def is_over(self):
        """
        Checks if the win condition is met and the game is over

        Returns:
        boolean: True if the win condition is met else False
        """
        return self.win()
    
    def show(self): 
        """
        Prints game informations after every move, current number and possible moves

        Returns:
        None
        """
        print (f"Current Number: {self.number}")
        print(f"Possible moves: {self.possible_moves()}") if self.possible_moves() else ""

    def scoring(self):
        """
        Gives a score to the current game (for the AI)

        Returns:
        int: Score value
        """
        return 100 if game.win() else 0

ai = Negamax(5) # The AI will think 5 moves in advance
game = divisorGame( 30, [ Human_Player(), AI_Player(ai) ] ) #Create game class that starts with 30 as initial value and 2 players (Human and AI)
history = game.play() # start game

print("\n--- GAME OVER ---")
print("You won!" if game.current_player != 1 else "AI won!")
