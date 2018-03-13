class TicTacToe:
    """Creating an instance of this class will train (if episodes>0) a
    reinforcement learning algorithm to play tic tac toe, and launch a simple GUI
    so the user can play against the algorithm.

    Setting episodes=0 means that the algorithm does not train, and it will make
    completely random moves. During every episode of training, the algorithm will
    literally play a game of tic tac toe against itself. It is rewarded for winning and
    penalized for losing, and it updates the value of observed states accordingly.
    I find that after around episodes=15000, the algorithm becomes unbeatable.

    If the user plays optimally, every game will therefore end in a draw.
    
    The GUI is very simple, as this was not the focus here.

    Author: Jakob Lorenzen"""
    
    def __init__(self, episodes=0, alpha=0.1, explore_exploit_ratio=0.1):
        #Train bot:
        #Explore-exploit ratio:
        self.explore_exploit_ratio = explore_exploit_ratio
        #Parameter that determines the update rate:
        self.alpha = alpha
        #Start with an empty board:
        self.first_state = "X|---------"
        #Dictionary to save the values for every observed state:
        self.states_dict = { self.first_state : 0.0 }

        #Train:
        for _ in range(episodes):
            self.train(self.first_state)
        
        #Launch GUI and play bot:
        import tkinter as tk
        #Current state
        self.current_state = self.first_state
        #winner holds the if, if one is found:
        self.winner = "none"
        #Run Tkinter
        self.root = tk.Tk()
        self.root.title("Tic Tac Toe")
        #Initialize is the main GUI command:
        self.initialize()
        self.root.mainloop()

    def train(self, state):
        #Find possible next states:
        possible_states = self.find_possible_states(state)
        
        #Pick a random number, to determine if we explore or exploit:
        import numpy as np
        rand = np.random.uniform()
        if rand > self.explore_exploit_ratio:
            #Exploit:
            #Find the optimal next state and it's value:
            future_state, future_value = self.exploit(state, possible_states)
            #Update the current state using the next value:
            self.update(state, future_value)

            #Check if future state implies a winner, and if it does,
            #we also update the value of the future state accordingly,
            #and then finish this episode by returning
            winner = self.is_winner(future_state)
            if winner=="X":
                #Bot is "O", so we penalize by -1
                reward = -1.0
                self.update(future_state, reward)
                return
            elif winner=="O":
                #Reward bot for winning
                reward = 1.0
                self.update(future_state, reward)
                return
            if self.is_final(future_state):
                #If board is full with no winner, we just end episode
                return 
        else:
            #Explore (pick random future state):
            future_state = self.explore(possible_states)
            
            #Check if future state implies a winner, and if it does,
            #we update the value of the future state accordingly,
            #and then finish this episode by returning
            winner = self.is_winner(future_state)
            if winner=="X":
                reward = -1.0
                self.update(future_state, reward)
                return
            elif winner=="O":
                reward = 1.0
                self.update(future_state, reward)
                return
            if self.is_final(future_state):
                return
        #If we haven't returned at this stage, we go to the next move of the episode.
        self.train(future_state)
        return

    def is_final(self, state):
        #Check if board is full
        if "-" in state:
            return False
        else:
            return True

    def is_winner(self, state):
        #Check if we have a winner
        if (state[2:5]=="XXX" or state[5:8]=="XXX" or state[8:11]=="XXX" or
            state[2]+state[5]+state[8] == "XXX" or state[3]+state[6]+state[9]=="XXX" or
            state[4]+state[7]+state[10]=="XXX" or
            state[2]+state[6]+state[10]=="XXX" or state[4]+state[6]+state[8]=="XXX"):
            return "X"
        elif (state[2:5]=="OOO" or state[5:8]=="OOO" or state[8:11]=="OOO" or
            state[2]+state[5]+state[8] == "OOO" or state[3]+state[6]+state[9]=="OOO" or
            state[4]+state[7]+state[10]=="OOO" or
            state[2]+state[6]+state[10]=="OOO" or state[4]+state[6]+state[8]=="OOO"):
            return "O"
        else:
            #No winner (but episode might not be over)
            return " "


    def find_possible_states(self, state):
        #States take the form like X|----X---O, where the first X means that it is X's turn.
        #Player whose turn it is
        player = state[0]
        next_player = "X" if player=="O" else "O"
        #new states holds possible future states
        new_states = []
        for position in range(2,11):
            new_state = next_player + state[1:]
            if new_state[position] == "-":
                new_state = new_state[:position] + player + new_state[position+1:]
                new_states.append(new_state)
        return new_states

    def exploit(self, state, states):
        #Exploit. Player "O" will try to maximize the value, while
        #player "X" will try to minimize. Therefore, look for the highest and lowest value:

        #First state is both the highest and lowest so far.
        #We put them in a list, as there may be multiple states with the same value.
        #(Later we choose randomly among them.
        high_state = [states[0]]
        low_state = [states[0]]

        #If state is not in dictionary (if it has not been encountered before), we add it with value=0.0:
        if states[0] not in  self.states_dict:
            self.states_dict[states[0]] = 0.0

        #Find best and worst states and values:
        high_value = self.states_dict[states[0]]
        low_value = self.states_dict[states[0]]
        for i in range(1, len(states)):
            if states[i] not in self.states_dict:
                self.states_dict[states[i]] = 0.0
            if self.states_dict[states[i]] > high_value:
                high_value = self.states_dict[states[i]]
                high_state = [states[i]]
            elif self.states_dict[states[i]] == high_value:
               high_value = self.states_dict[states[i]]
               high_state.append(states[i])
            if self.states_dict[states[i]] < low_value:
                low_value = self.states_dict[states[i]]
                low_state = [states[i]]
            elif self.states_dict[states[i]] == low_value:
               low_value = self.states_dict[states[i]]
               low_state.append(states[i])
               
        #Choose randomly among states:
        import numpy as np
        #If Player is "O", we need the state with the highest value
        if state[0]=='O':
            choice = np.random.choice(len(high_state))
            high_state = high_state[choice]
            return high_state, high_value
        else:
            #For player "X", we need the state with the lowest value
            choice = np.random.choice(len(low_state))
            low_state = low_state[choice]
            return low_state, low_value            

    def explore(self, states):
        #Explore: Pick randomly among possible future states:
        import numpy as np
        choice = np.random.choice(len(states))
        if choice not in self.states_dict:
            self.states_dict[states[choice]] = 0.0
        return states[choice]

    def update(self, state, next_value):
        #Update rule for a state given a future value
        self.states_dict[state] += self.alpha * (next_value - self.states_dict[state])
        return

    def reset(self):
        #Reset current state and clickable buttons:
        self.current_state = self.first_state
        self.winner = "none"
        self.button0["text"] = " "
        self.button1["text"] = " "
        self.button2["text"] = " "
        self.button3["text"] = " "
        self.button4["text"] = " "
        self.button5["text"] = " "
        self.button6["text"] = " "
        self.button7["text"] = " "
        self.button8["text"] = " "
        return

    def initialize(self):
        #Initialize the GUI with 9 buttons
        import tkinter as tk
        import numpy  as np
        height = 2
        width = 8

        self.button0 = tk.Button(self.root, text=" ", font=('Times 26 bold'), height=height, width=width, command=lambda: self.click(self.button0, 0))
        self.button0.grid(row=0, column=0, sticky = tk.S + tk.N + tk.E + tk.W)

        self.button1 = tk.Button(self.root, text=" ", font=('Times 26 bold'), height=height, width=width, command=lambda: self.click(self.button1, 1))
        self.button1.grid(row=0, column=1, sticky = tk.S + tk.N + tk.E + tk.W)

        self.button2 = tk.Button(self.root, text=" ", font=('Times 26 bold'), height=height, width=width, command=lambda: self.click(self.button2, 2))
        self.button2.grid(row=0, column=2, sticky = tk.S + tk.N + tk.E + tk.W)

        self.button3 = tk.Button(self.root, text=" ", font=('Times 26 bold'), height=height, width=width, command=lambda: self.click(self.button3, 3))
        self.button3.grid(row=1, column=0, sticky = tk.S + tk.N + tk.E + tk.W)

        self.button4 = tk.Button(self.root, text=" ", font=('Times 26 bold'), height=height, width=width, command=lambda: self.click(self.button4, 4))
        self.button4.grid(row=1, column=1, sticky = tk.S + tk.N + tk.E + tk.W)

        self.button5 = tk.Button(self.root, text=" ", font=('Times 26 bold'), height=height, width=width, command=lambda: self.click(self.button5, 5))
        self.button5.grid(row=1, column=2, sticky = tk.S + tk.N + tk.E + tk.W)

        self.button6 = tk.Button(self.root, text=" ", font=('Times 26 bold'), height=height, width=width, command=lambda: self.click(self.button6, 6))
        self.button6.grid(row=2, column=0, sticky = tk.S + tk.N + tk.E + tk.W)

        self.button7 = tk.Button(self.root, text=" ", font=('Times 26 bold'), height=height, width=width, command=lambda: self.click(self.button7, 7))
        self.button7.grid(row=2, column=1, sticky = tk.S + tk.N + tk.E + tk.W)

        self.button8 = tk.Button(self.root, text=" ", font=('Times 26 bold'), height=height, width=width, command=lambda: self.click(self.button8, 8))
        self.button8.grid(row=2, column=2, sticky = tk.S + tk.N + tk.E + tk.W)
        
        button_reset = tk.Button(self.root, text="Reset", font=('Times 26 bold'), height=height, width=width, command=self.reset)
        button_reset.grid(row=3, column=1, sticky = tk.S + tk.N + tk.E + tk.W)

    def click(self, button, position):
        #If button is not empty or a game is over, take no action:
        if button["text"] != " " or self.winner!="none":
            return
        #Else, fill button:
        self.fill(button, position)
        #Check if we have a winner
        self.check()
        #If there is no winner, bot makes a move:
        if self.winner=="none":
            self.bot_move()
            #Check for winner
            self.check()
        return

    def fill(self, button, position):
        #Fill button:
        #First character in the state is the player whose turn it is
        to_fill = self.current_state[0]
        next_player = "X" if to_fill=="O" else "O"
        #Update GUI button
        button["text"] = to_fill
        #Update internal state:
        self.current_state = next_player + self.current_state[1:2+position] + to_fill + self.current_state[3+position:]
            

    def check(self):
        #Check if we have a winner:
        winner = self.is_winner(self.current_state)
        if winner!=" ":
            self.winner = winner

        #If there is a winner or if game is over, show info box:
        import tkinter.messagebox
        if self.winner != "none":
            tkinter.messagebox.showinfo("Winner!", "Player " +self.winner+ ", you have just won the game")
        elif self.is_final(self.current_state):
            self.winner = "No winner"
            tkinter.messagebox.showinfo("Game over!", "All fields are occupied")
        return        
            
    def bot_move(self):
        #Bots turn to make a move.
        #Find possible states:
        possible_states = self.find_possible_states(self.current_state)
        #Choose a state (exploit):
        chosen_state, _ = self.exploit(self.current_state, possible_states)
        #List of all buttons:
        button_list = [self.button0, self.button1, self.button2, self.button3, self.button4,
                       self.button5, self.button6, self.button7, self.button8]
        #Run through buttons to find the one that needs updating:
        for i, symbol, button in zip(range(9), chosen_state[2:], button_list):
            if symbol != button["text"] and symbol != "-":
                position = i
        #Now that we found the button that needs updating, fill it:
        self.fill(button_list[position], position)
        return


        
TicTacToe(episodes=15000)
