# lucky_game.py

from __future__ import annotations
import re
import random
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, List


NAME_RE = re.compile(r"^[A-Za-zÅÄÖåäö]+ [A-Za-zÅÄÖåäö]+$")  # exactly one space, letters only


@dataclass(frozen=True)
class Birthdate:
    year: int
    month: int
    day: int

    def to_date(self) -> date:
        return date(self.year, self.month, self.day)


def validate_name(name: str) -> str:
    """Step 1: full name (letters only) with exactly one space."""
    if not isinstance(name, str) or not NAME_RE.fullmatch(name.strip()):
        raise ValueError("Name must be letters only with ONE space (e.g., 'Anna Karlsson').")
    return name.strip()


def parse_birthdate_raw(raw: str) -> Birthdate:
    """Step 2: birthdate as yyyymmdd (must be a real calendar date)."""
    if not (isinstance(raw, str) and raw.isdigit() and len(raw) == 8):
        raise ValueError("Birthdate must be 8 digits in yyyymmdd format.")
    y, m, d = int(raw[:4]), int(raw[4:6]), int(raw[6:8])
    bd = Birthdate(y, m, d)
    # raise if invalid (e.g., 19990230)
    bd.to_date()
    return bd


def compute_age(bd: Birthdate, as_of_year: int = 2022) -> int:
    """Step 3/4: age computed relative to end of year 2022 (per assignment)."""
    b = bd.to_date()
    ref = date(as_of_year, 12, 31)
    return ref.year - b.year - ((ref.month, ref.day) < (b.month, b.day))


def require_adult(age: int) -> None:
    """Step 4: must be 18+."""
    if age < 18:
        raise ValueError("Player must be 18 or older.")


def validate_choice_in_list(value: str, options: List[int]) -> int:
    """Ensure the input is an int and is one of the offered options."""
    if not value.strip().lstrip("-").isdigit():
        raise ValueError("Input must be an integer.")
    num = int(value)
    if num not in options:
        raise ValueError("Pick a number from the displayed list.")
    return num



@dataclass
class Player:
    player_name: str | None = None
    player_birthdate: Birthdate | None = None
    player_age: int | None = None

    def set_name(self, name: str) -> None:
        self.player_name = validate_name(name)

    def set_birthdate_from_raw(self, raw: str) -> None:
        self.player_birthdate = parse_birthdate_raw(raw)
        self.player_age = compute_age(self.player_birthdate)

    def ensure_adult(self) -> None:
        if self.player_age is None:
            raise ValueError("Age not computed yet.")
        require_adult(self.player_age)


@dataclass
class CPU:
    lucky_list: List[int] = field(default_factory=list)
    lucky_number: int | None = None
    rng: random.Random = field(default_factory=lambda: random.Random())

    # Step 5
    def make_lucky_list(self, n: int = 9, lo: int = 0, hi: int = 100) -> List[int]:
        # unique numbers to avoid duplicates confusion
        self.lucky_list = sorted(self.rng.sample(range(lo, hi + 1), k=n))
        return self.lucky_list

    # Step 6
    def pick_lucky_number(self, lo: int = 0, hi: int = 100) -> int:
        self.lucky_number = self.rng.randint(lo, hi)
        return self.lucky_number

    # Step 6 (append to make list of 10)
    def extend_with_lucky_number(self) -> List[int]:
        if self.lucky_number is None:
            raise ValueError("lucky_number not set.")
        if self.lucky_number not in self.lucky_list:
            self.lucky_list.append(self.lucky_number)
            self.lucky_list.sort()
        # if already in list, add a random unique number to keep total at 10
        while len(self.lucky_list) < 10:
            candidate = self.rng.randint(0, 100)
            if candidate not in self.lucky_list:
                self.lucky_list.append(candidate)
                self.lucky_list.sort()
        return self.lucky_list

    # Step 9
    def shorten_list_around_center(self, center: int, radius: int = 10) -> List[int]:
        lo, hi = max(0, center - radius), min(100, center + radius)
        return [x for x in self.lucky_list if lo <= x <= hi]



InputFn = Callable[[str], str]
PrintFn = Callable[[str], None]


@dataclass
class Game:
    player: Player = field(default_factory=Player)
    cpu: CPU = field(default_factory=CPU)
    tries_count: int = 0
    player_input: int | None = None
    shorter_lucky_list: List[int] = field(default_factory=list)
    ask: InputFn = input
    say: PrintFn = print

    # Steps 1–4
    def onboarding(self) -> None:
        # 1) Name
        while True:
            try:
                self.player.set_name(self.ask("Enter your full name (First Last): "))
                break
            except Exception as e:
                self.say(f"Invalid name: {e}")
        # 2–4) Birthdate -> Age -> 18+
        while True:
            try:
                raw = self.ask("Enter your birthdate as yyyymmdd: ")
                self.player.set_birthdate_from_raw(raw)
                self.player.ensure_adult()
                self.say(f"Your age in 2022 is {self.player.player_age}.")
                break
            except Exception as e:
                self.say(f"Invalid birthdate/age: {e} — try again.")

    # Steps 5–7
    def prepare_round(self) -> None:
        self.tries_count = 0
        self.cpu.make_lucky_list(n=9)
        self.cpu.pick_lucky_number()
        self.cpu.extend_with_lucky_number()
        self.say(f"\nLucky list (10 numbers 0–100): {self.cpu.lucky_list}")

    def prompt_guess_from_list(self, current_list: List[int]) -> int:
        while True:
            try:
                return validate_choice_in_list(self.ask("Pick your lucky number from the list: "), current_list)
            except Exception as e:
                self.say(f"Invalid choice: {e}")

    # Steps 8–12
    def handle_guess(self, guess: int) -> bool:
        self.tries_count += 1
        self.player_input = guess
        if guess == self.cpu.lucky_number:
            self.say(f"Congrats, game is over! You got lucky number from try #{self.tries_count} 🎉")
            return True

        # Wrong guess -> build shorter list within ±10 of lucky_number
        self.shorter_lucky_list = self.cpu.shorten_list_around_center(self.cpu.lucky_number, radius=10)
        # Step 12: delete the wrong guess from the shorter list if present
        if guess in self.shorter_lucky_list:
            self.shorter_lucky_list = [x for x in self.shorter_lucky_list if x != guess]

        self.say(f"this is try #{self.tries_count} and new list is: {self.shorter_lucky_list}")
        return False

    # Step 13
    def should_end(self) -> bool:
        return len(self.shorter_lucky_list) <= 2

    def play_once(self) -> None:
        self.prepare_round()
        # First guess on the full list
        guess = self.prompt_guess_from_list(self.cpu.lucky_list)
        if self.handle_guess(guess):
            return
        # Keep asking on shorter lists until guessed or only 2 left
        while not self.should_end():
            guess = self.prompt_guess_from_list(self.shorter_lucky_list)
            if self.handle_guess(guess):
                return
        self.say("Game ended: list is down to 2 numbers. Better luck next time!")

    def loop(self) -> None:
        self.onboarding()
        while True:
            self.play_once()
            again = self.ask("Do you like to play again? (y=yes, n=no): ").strip().lower()
            if again != "y":
                self.say("Goodbye!")
                break


def main() -> None:
    Game().loop()


if __name__ == "__main__":
    main()
