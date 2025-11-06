r"""
Projekt: Fuzzy Acrobot (Acrobot-v1, Gymnasium + scikit-fuzzy)

Autorzy: Adrian Kopczyński, Gabriel Francke

Instrukcja środowiska:
  pip install -r requirements.txt
  pip install -r .\NAI_02_Fuzzy_Logic\requirements.txt

Uruchomienie:
  python game.py

Opis problemu:
  Sterujemy dwuczłonowym wahadłem (Acrobot), którego celem jest wybicie końcówki ponad linię celu.
  Stan środowiska (obs) = [cosθ1, sinθ1, cosθ2, sinθ2, θ̇1, θ̇2].

  Logika rozmyta wykorzystuje 3 wejścia:
    1) theta2_abs  – bezwzględna wartość kąta drugiego członu, w zakresie [0, π]
    2) omega_sum   – suma prędkości kątowych (θ̇1 + θ̇2), typowo ~[-8, 8]
    3) energy_err  – błąd energetyczny, wyznaczany na podstawie potencjalnej i kinetycznej energii
                     (dodatni → niedobór energii, ujemny → nadmiar energii)

    Wyjście FIS:
    - torque w [-1, 1] → zamiar momentu siły, który po przejściu przez histerezę
      jest kwantyzowany do akcji dyskretnych {-1, 0, +1}.

Uwaga:
  - System używa automatycznych funkcji przynależności (automf) dla uproszczenia strojenia.
  - Zaimplementowano prostą histerezę, aby uniknąć oscylacji między akcjami.
  - Parametry alpha, beta i scale sterują proporcją między energią potencjalną a kinetyczną.
  - Wyniki epizodów wypisywane są w konsoli (total_reward).
"""

from __future__ import annotations
import math
import time
from typing import Tuple

import gymnasium as gym

import numpy as np

import skfuzzy as fuzz
from skfuzzy import control as ctrl


# ---------- Pomocnicze przekształcenia stanu ----------
def extract_angles(obs: np.ndarray) -> Tuple[float, float]:
    """Zwraca (theta1, theta2) z wektora [cosθ1, sinθ1, cosθ2, sinθ2, θ̇1, θ̇2]."""
    c1, s1, c2, s2 = obs[0], obs[1], obs[2], obs[3]
    theta1 = math.atan2(s1, c1)
    theta2 = math.atan2(s2, c2)
    return theta1, theta2

def energy_deficit(theta1: float, theta2: float, omega1: float, omega2: float,
                   alpha: float = 1.0, beta: float = 0.20, scale: float = 1.0) -> float:
    """
    Oblicza błąd energii układu (energy_err), czyli różnicę między potrzebną a aktualną energią wahadła.

    Działanie:
      - Gdy zwracana wartość jest dodatnia → brakuje energii, wahadło trzeba rozpędzać.
      - Gdy wartość jest ujemna → energii jest za dużo, należy hamować.
      - Wynik jest ograniczony do zakresu [-1, 1].

    Parametry:
      theta1, theta2: kąty obu ramion wahadła
      omega1, omega2: prędkości kątowe (czyli jak szybko poruszają się ramiona)
      alpha: jak mocno liczy się energia potencjalna (czyli wysokość wahadła)
      beta: jak mocno liczy się energia kinetyczna (czyli prędkość ruchu)
      scale: współczynnik skalujący, który dopasowuje wynik do [-1, 1]

    Zwraca:
      Wartość błędu energii (float) – używana później jako jedno z wejść systemu rozmytego.
    """
    y = math.sin(theta1) + math.sin(theta1 + theta2)  # [-2, 2]
    E_pot = (1.0 - y)                                  # 0 przy celu ~y=1
    E_kin = 0.5 * (omega1*omega1 + omega2*omega2)

    err = alpha * E_pot - beta * E_kin
    return max(-1.0, min(1.0, err / scale))


# ---------- Definicja układu rozmytego (3 wejścia → 1 wyjście) ----------
def build_fuzzy_controller() -> ctrl.ControlSystemSimulation:
    """
    Tworzy rozmyty kontroler (Fuzzy Logic Controller) sterujący ruchem wahadła Acrobot.

    Wejścia:
      - theta2_abs : kąt drugiego ramienia, mówi jak bardzo wahadło jest odchylone
      - omega_sum  : suma prędkości obrotowych obu ramion
      - energy_err : błąd energii mówi, czy brakuje energii (wartość dodatnia) lub jest jej za dużo (ujemna)

    Wyjście:
      - torque : moment siły w zakresie od -1 do 1, który później zamieniany jest na akcję {-1, 0, +1}

    Jak to działa:
      - Dla każdego wejścia tworzone są automatyczne strefy (mała, średnia, duża wartość).
      - Reguły decydują, kiedy dodawać energię (pompowanie), a kiedy hamować.

    Zwraca:
      Gotowy kontroler, który można użyć w pętli głównej programu.
    """
    # --- Zakresy ---
    theta2_abs = ctrl.Antecedent(np.linspace(0, math.pi, 101), 'theta2_abs')
    omega_sum  = ctrl.Antecedent(np.linspace(-8, 8, 161), 'omega_sum')
    energy_err = ctrl.Antecedent(np.linspace(-1, 1, 201), 'energy_err')
    torque     = ctrl.Consequent(np.linspace(-1, 1, 201), 'torque')

    # --- Auto MF ---
    # automf(3) tworzy 3 etykiety/poziomy: 'poor', 'average', 'good'

    # theta2_abs: mały kąt = poor, duży kąt = good
    theta2_abs.automf(3)

    # omega_sum: ujemna prędkość = poor, dodatnia = good
    omega_sum.automf(3)

    # energy_err: brak energii = good, nadmiar = poor
    energy_err.automf(3)

    # torque: moment w lewo = poor, brak = average, w prawo = good
    torque.automf(3)

    # --- Reguły ---
    rules = [
        # Jeśli brakuje energii i prędkość jest dodatnia – daj moment w tym samym kierunku (pompowanie)
        ctrl.Rule(energy_err['good'] & omega_sum['good'], torque['good']),

        # Jeśli brakuje energii, ale prędkość ujemna – daj moment w przeciwnym kierunku (też pompowanie)
        ctrl.Rule(energy_err['good'] & omega_sum['poor'], torque['poor']),

        # Jeśli mamy za dużo energii i prędkość dodatnia – hamuj (moment przeciwny do ruchu)
        ctrl.Rule(energy_err['poor'] & omega_sum['good'], torque['poor']),

        # Jeśli mamy za dużo energii i prędkość ujemna – również hamuj w przeciwnym kierunku
        ctrl.Rule(energy_err['poor'] & omega_sum['poor'], torque['good']),

        # Jeśli energia i prędkość są średnie – nie podejmuj akcji (utrzymuj stan)
        ctrl.Rule(energy_err['average'] & omega_sum['average'], torque['average']),

        # Jeśli wahadło jest blisko dolnej pozycji i brakuje energii – „kopnij” w odpowiednią stronę
        ctrl.Rule(energy_err['good'] & omega_sum['good'] & theta2_abs['poor'], torque['good']),
        ctrl.Rule(energy_err['good'] & omega_sum['poor'] & theta2_abs['poor'], torque['poor']),
    ]

    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)

    # Defuzyfikacja: sposób zamiany wyniku rozmytego na liczbę
    # Testowałem różne warianty:
    # 1) 'mom' (mean of maxima) – wybiera średnią z wartości o maksymalnej przynależności.
    #    Efekt: model reagował wolno i po chwili tracił momentum wahadła.
    #
    # 2) 'som' (smallest of maxima) – wybiera pierwszy (najmniejszy) punkt maksimum.
    #    Efekt: częste oscylacje i brak energii do wybicia wahadła.
    #
    # 3) 'lom' (last of maxima) – wybiera ostatni (największy) punkt maksimum.
    #    Efekt: bardziej zdecydowane ruchy, szybsze pompowanie energii i stabilniejsze wybicia.
    #
    # Dlatego obecnie używana metoda to 'lom', która daje najlepszy kompromis między płynnością a siłą ruchu.
    torque.defuzzify_method = 'lom'
    return sim

def quantize_action_hyst(tintent: float, prev: int, tau_on: float = 0.12, tau_off: float = 0.05) -> int:
    """
    Zamienia ciągly sygnal momentu (torque_intent) na dyskretna akcje {-1, 0, +1}
    z wykorzystaniem histerezy, aby uniknąć szybkich zmian kierunku ruchu.

    Działanie:
      - jeśli torque jest mały, akcja pozostaje 0
      - jeśli przekroczy próg tau_on, rozpoczyna ruch w danym kierunku
      - ruch trwa, dopóki torque nie spadnie poniżej tau_off
    """
    # Gdy obecnie stoimy (0), włącz ruch dopiero po przekroczeniu wyższego progu
    if prev == 0:
        if tintent >  tau_on: return +1 # start ruchu w prawo
        if tintent < -tau_on: return -1 # start ruchu w lewo
        return 0                        # stoje
    
    # Gdy jedziemy w prawo (+1), trzymaj kierunek dopóki sygnał nie spadnie poniżej niższego progu
    elif prev == +1:
        return +1 if tintent >  tau_off else 0
    
    # Gdy jedziemy w lewo (-1), analogicznie
    else:
        return -1 if tintent < -tau_off else 0

# ---------- Pętla epizodów ----------
def run(
    seed: int = 42,
    episodes: int = 5,
    render_mode: str | None = "human",      # None = szybciej; "human" = podgląd
    tau_on: float = 0.05,                 # próg załączenia histerezy
    tau_off: float = 0.02,                # próg wyłączenia histerezy
    alpha: float = 1.0,                   # waga energii potencjalnej w energy_deficit
    beta: float = 0.15,                   # waga energii kinetycznej w energy_deficit
    scale: float = 0.8,                   # normalizacja energy_err (mniejsze = bardziej czuły)
    debug_every: int = 200                # co ile kroków wypisać debug; <=0 wyłącza
):
    """
    Uruchamia kilka gier (epizodów) Acrobota sterowanych rozmytym kontrolerem (Fuzzy Logic).
    Po każdym epizodzie wypisuje łączny wynik (total_reward).

    Jak działa punktacja:
    - W Acrobot-v1 każda klatka daje -1 punkt.
    - Im szybciej wahadło osiągnie cel (mniej kroków), tym lepszy wynik.
    - Wynik -500 oznacza, że nie udało się osiągnąć celu w czasie limitu.

    Najważniejsze parametry:
    - render_mode: None = szybkie uruchomienie, "human" = pokazuje animację.
    - tau_on / tau_off: progi histerezy (sterują, kiedy zaczyna i kończy się ruch).
    - alpha / beta: określają, jak ważna jest energia potencjalna i kinetyczna.
    - scale: ustala czułość błędu energii (mniejsza wartość = silniejsza reakcja).
    - debug_every: co ile kroków wypisać podgląd danych (0 = brak logów).

    Wypisywane informacje:
    - Po każdym epizodzie: całkowita nagroda (total_reward).
    - Opcjonalnie: co kilka kroków podgląd wartości wejściowych i wyjścia kontrolera.
    """

    # --- Utwórz środowisko (bez renderu jest dużo szybciej) ---
    env = gym.make("Acrobot-v1", render_mode=("human" if render_mode == "human" else None))
    env.reset(seed=seed)

    # --- Rozmyty kontroler (FIS) ---
    sim = build_fuzzy_controller()

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0             # akumuluje -1 za krok
        prev_action_cont = 0           # pamięć histerezy w skali ciągłej (-1..+1)
        step = 0

        while not done:
            # --- Wejścia do kontrolera ---
            theta1, theta2_v = extract_angles(obs)
            theta2_abs_v = abs(theta2_v)
            omega1, omega2 = obs[4], obs[5]
            omega_sum_v = omega1 + omega2

            # Błąd energii (używamy parametrów z funkcji, nie na sztywno)
            energy_err_v = energy_deficit(theta1, theta2_v, omega1, omega2,
                                          alpha=alpha, beta=beta, scale=scale)

            # --- FIS: wyznacz zamiar momentu ---
            sim.reset()
            sim.input['theta2_abs'] = theta2_abs_v
            sim.input['omega_sum']  = omega_sum_v
            sim.input['energy_err'] = energy_err_v
            sim.compute()
            torque_intent = float(sim.output['torque'])  # [-1, 1]

            # --- Histereza → akcja ciągła {-1,0,+1} ---
            raw = quantize_action_hyst(torque_intent, prev_action_cont, tau_on=tau_on, tau_off=tau_off)
            prev_action_cont = raw

            # --- MAPOWANIE na kody akcji Acrobota {0,1,2} ---
            # 0: moment ujemny, 1: zero, 2: moment dodatni
            action = {-1: 0, 0: 1, 1: 2}[int(raw)]

            # --- Krok środowiska ---
            obs, reward, terminated, truncated, info = env.step(action)
            done = (terminated or truncated)
            total_reward += reward

            # --- Debug co N kroków (włącz/wyłącz parametrem debug_every) ---
            if debug_every and debug_every > 0 and (step % debug_every == 0):
                print(f"s={step} | θ2_abs={theta2_abs_v:.2f}, ω_sum={omega_sum_v:.2f}, "
                      f"E_err={energy_err_v:.2f}, torque={torque_intent:.2f}, a={action}")

            step += 1

        print(f"[Episode {ep+1}] total_reward={total_reward:.1f}")

    env.close()


if __name__ == "__main__":
    run()
