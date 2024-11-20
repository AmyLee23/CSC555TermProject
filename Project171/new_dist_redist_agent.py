"""
**Submitted to ANAC 2024 SCML (OneShot track)**
*Authors*
        Name : Hajime Endo (20213)
        Nationality : Japan
        Email : endo@katfuji.lab.tuat.ac.jp
        Univ : Tokyo University of Agriculture and Technology

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML.

I would like to extend my heartfelt gratitude to everyone involved in the organization of ANAC 2024,
as well as to all the participants who competed alongside me. Thank you.

"""

from __future__ import annotations
from negmas import SAOResponse, ResponseType, SAOState
from negmas.sao import SAONMI
from scml.oneshot import *
from scml.oneshot.ufun import *
import random
from typing import Any
from scml.oneshot import OneShotSyncAgent
from negmas import Contract
from itertools import chain, combinations
import math

__all__ = ["NewDistRedistAgent"]


def distribute(q: int, n: int) -> list[int]:
    """Distributes n values over m bins with at
    least one item per bin assuming q > n"""
    from numpy.random import choice
    from collections import Counter

    if n == 0:
        return []

    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst

    if q == n:
        return [1] * n
    r = Counter(choice(n, q - n))
    return [r.get(_, 0) + 1 for _ in range(n)]


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class NewDistRedistAgent(OneShotSyncAgent):
    round: int
    random_p: float
    all_partners_info = dict()  # adding a new field to this dict

    def __init__(self):
        self.random_p = 0.07
        self.adaptive_threshold = 0.5  # Threshold for adapting randomness
        self.decay_factor = 0.9  # Decay factor for historical data (e.g., 0.9 means a 10% decay per step)
        self.seasonal_amplitude = 0.2  # Amplitude of the seasonal effect (e.g., 20% of base price)
        self.seasonal_frequency = 2 * math.pi / 365  # Frequency for a yearly cycle (2π / 365 days)
        self.seasonal_phase_shift = 0  # Phase shift to align with the desired start of the cycle
        super().__init__()

    def update_randomness(self):
        """
        Adjusts random_p based on agent performance and negotiation progress.
        """
        num_partners = len(self.all_partners_info)
        self.adaptive_threshold = max(0.3, min(0.7, num_partners / 10))
        
        # Calculate overall success rate
        total_contracts = sum(
            info[0] + info[2] for info in self.all_partners_info.values()
        )
        
        if total_contracts == 0:
            # Use higher randomness initially for exploration
            self.random_p = 0.2
            return

        successful_contracts = sum(info[0] for info in self.all_partners_info.values())
        success_rate = successful_contracts / total_contracts

        # Adjust randomness based on success rate
        if success_rate < self.adaptive_threshold:
            # Higher randomness for lower success rates
            self.random_p = min(0.5, self.random_p + 0.05)
        else:
            # Lower randomness for higher success rates
            self.random_p = max(0.05, self.random_p - 0.05)

        # Optional: Adjust based on current negotiation step
        step_factor = max(0.1, 1 - (self.awi.current_step / 50))  # Decreases over 50 steps
        self.random_p *= step_factor


    def distribute_needs_randomly(self, needs, partner_ids) -> dict[str, int]:
        self.update_randomness()  # Update randomness before distribution
        dist = dict()
        partners = len(partner_ids)
        if needs <= 0:
            dist.update(dict(zip(partner_ids, [0] * partners)))
            return dist

        # Scale needs with randomness bias
        adjusted_needs = int(needs * (1 + (random.random() - 0.5) * self.random_p))
        dist.update(dict(zip(partner_ids, distribute(adjusted_needs, partners))))
        return dist
    
    # Calculate a priority score based on exponential decay (10%)
    def calc_weighted_score(self, partners) :
        for p in partners: 
            s_contracts = self.all_partners_info[p][0]
            total_quantity = self.all_partners_info[p][1]
            u_contracts = self.all_partners_info[p][2]
            contractAge = self.all_partners_info[p][3]

            # Apply the decay factor to adjust the weight of older data
            ds_contracts = s_contracts * (self.decay_factor ** contractAge)
            decayed_total_quantity = total_quantity * (self.decay_factor ** contractAge)
            du_contracts = u_contracts * (self.decay_factor ** contractAge)
            totald_contracts = ds_contracts + du_contracts

            # Calculate average quantity and success rate using decayed values
            if (totald_contracts) == 0 or ds_contracts == 0:
                return 0

            average_quantity = decayed_total_quantity / ds_contracts    # this is the important change 
            success_rate = ds_contracts / totald_contracts

            priority_score = average_quantity * success_rate
        return priority_score

    # original priority score calculation, without decay factor 
    def get_partners_list(self, partners) :
        partner_list = []

        for p in partners:
            s_contracts = self.all_partners_info[p][0]
            total_quantity = self.all_partners_info[p][1]
            u_contracts = self.all_partners_info[p][2]
            total_contracts = s_contracts + u_contracts

            # Check if the denominator is not zero to avoid division errors
            if total_contracts != 0 and s_contracts != 0:
                avg_quantity = total_quantity / s_contracts
                success_rate = s_contracts / total_contracts
                priority_score = avg_quantity * success_rate
            else:
                priority_score = 0

        # Append the partner, priority score, and initial allocation value
        partner_list.append([p, priority_score, 0])
        return partner_list
        

    def distribute_by_info(self, need, partners):
        dist = dict()
        if len(partners) == 0:
            return dist

        if need <= 0:
            dist.update(dict(zip(partners, [0] * len(partners))))

        #partners_list = self.get_partners_list(partners)

        # Calculate priority scores for each partner
        partner_list = [
            [partner, self.calc_weighted_score(partner), 0] for partner in partners
        ]

        priority_list = sorted(partner_list, key=lambda x: x[1], reverse=True)

        sum = 0
        for i in range(len(priority_list)):
            sum += priority_list[i][1]

        if sum == 0:
            sum = 1
        p = 0
        prob_list = []
        for i in range(len(priority_list)):
            prob_list.append((p, p + priority_list[i][1] / sum))
            p += priority_list[i][1] / sum

        for _ in range(need):
            r = random.random()
            for i in range(len(prob_list)):
                if prob_list[i][0] <= r < prob_list[i][1]:
                    priority_list[i][2] += 1
                    break

        # Force allocation of 1 or more
        for i in range(len(priority_list)):
            if priority_list[i][2] == 0:
                priority_list[i][2] = 1

        dist_list = []
        for k in partners:
            for l in priority_list:
                if k == l[0]:
                    dist_list.append(l[2])

        dist.update(dict(zip(partners, dist_list)))
        return dist

    def distribute_by_offers(self, need, partners, offers):
        dist = dict()

        if len(partners) == 0:
            return dist

        if need <= 0:
            dist.update(dict(zip(partners, [0] * len(partners))))
            return dist

        partner_list = [[partner, offers[partner][QUANTITY], 0] for partner in partners]
        # partner_list = [[partner,
        #                ((self.all_partners_info[partner][1]/self.all_partners_info[partner][0]) * (self.all_partners_info[partner][0]/(self.all_partners_info[partner][3] + self.all_partners_info[partner][0]))) if (self.all_partners_info[partner][0] + self.all_partners_info[partner][3]) != 0 and self.all_partners_info[partner][0] != 0 else 0,
        #                0] for partner in partners]
        priority_list = sorted(partner_list, key=lambda x: x[1], reverse=True)

        sum = 0
        for i in range(len(priority_list)):
            sum += priority_list[i][1]

        if sum == 0:
            sum = 1
        p = 0
        prob_list = []
        for i in range(len(priority_list)):
            prob_list.append((p, p + priority_list[i][1] / sum))
            p += priority_list[i][1] / sum

        for _ in range(need):
            r = random.random()
            for i in range(len(prob_list)):
                if prob_list[i][0] <= r < prob_list[i][1]:
                    priority_list[i][2] += 1
                    break

        # Force allocation of 1 or more
        for i in range(len(priority_list)):
            if priority_list[i][2] == 0:
                priority_list[i][2] = 1

        dist_list = []
        for k in partners:
            for l in priority_list:
                if k == l[0]:
                    dist_list.append(l[2])

            # distribute my needs over my (remaining) partners.
        dist.update(dict(zip(partners, dist_list)))
        return dist

    def first_proposals(self):
        if self.awi.is_first_level:
            partners = self.awi.my_consumers
            need = self.awi.needed_sales
        else:
            partners = self.awi.my_suppliers
            need = self.awi.needed_supplies

        s, p = self._step_and_price(best_price=True)

        if self.awi.current_step < 10 or random.random() < self.random_p:
            distribution = self.distribute_needs_randomly(need, partners)
        else:
            distribution = self.distribute_by_info(need, partners)

        d = {k: (q, s, p) if q > 0 else None for k, q in distribution.items()}

        return d

    def counter_all(self, offers, states):
        self.round += 1
        response = dict()

        # process for sales and supplies independently
        for needs, all_partners, issues in [
            (
                self.awi.needed_supplies,
                self.awi.my_suppliers,
                self.awi.current_input_issues,
            ),
            (
                self.awi.needed_sales,
                self.awi.my_consumers,
                self.awi.current_output_issues,
            ),
        ]:
            # get a random price
            price = issues[UNIT_PRICE].rand()

            # find active partners
            partners = {_ for _ in all_partners if _ in offers.keys()}
            plist = list(powerset(partners))

            offers_list = []
            for i, partner_ids in enumerate(plist):
                others = partners.difference(partner_ids)
                offered = sum(offers[p][QUANTITY] for p in partner_ids)

                offer_list = [offers[p] for p in partner_ids]

                utility = self.ufun.from_offers(
                    tuple(offer_list),
                    tuple([True for _ in range(len(offer_list))]),
                    ignore_signed_contracts=False,
                )

                offers_list.append((offered, partner_ids, others, utility))

            sorted_offers_list = sorted(offers_list, key=lambda x: x[3], reverse=True)
            th = 0
            th_round = 16
            th_over = 2

            for i, (offered, partner_ids, others, utility) in enumerate(
                sorted_offers_list
            ):
                # Redistribute when the 'for' loop is about to finish
                if i == len(sorted_offers_list) - 1 and needs != 0:
                    if random.random() < self.random_p:
                        distribution = self.distribute_needs_randomly(
                            needs, partner_ids
                        )
                    else:
                        distribution = self.distribute_by_offers(
                            needs, partner_ids, offers
                        )

                    response.update(
                        {
                            k: SAOResponse(ResponseType.REJECT_OFFER, None)
                            if q == 0
                            else SAOResponse(
                                ResponseType.REJECT_OFFER,
                                (q, self.awi.current_step, price),
                            )
                            for k, q in distribution.items()
                        }
                    )
                    break

                if needs > offered:
                    if offered >= th:
                        others = list(others)

                        if len(others) == 0 and needs - offered >= th_over:
                            continue

                        if random.random() < self.random_p:
                            remaining_dist = self.distribute_needs_randomly(
                                needs - offered, others
                            )
                        else:
                            remaining_dist = self.distribute_by_offers(
                                needs - offered, others, offers
                            )

                        response |= {
                            k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                            for k in partner_ids
                        } | {
                            k: SAOResponse(
                                ResponseType.REJECT_OFFER,
                                (q, self.awi.current_step, price),
                            )
                            for k, q in remaining_dist.items()
                        }
                        break
                    else:
                        continue

                elif needs < offered:
                    if self.round <= th_round:
                        continue
                    else:
                        # accept
                        if offered - needs <= th_over:
                            others = list(others)
                            response |= {
                                k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                                for k in partner_ids
                            } | {
                                k: SAOResponse(ResponseType.END_NEGOTIATION, None)
                                for k in others
                            }
                            break
                        else:
                            continue

                elif needs == offered:
                    others = list(others)
                    response |= {
                        k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                        for k in partner_ids
                    } | {
                        k: SAOResponse(ResponseType.END_NEGOTIATION, None)
                        for k in others
                    }
                    break

        return response

    def _step_and_price(self, best_price=False):
        """Returns current step and a random (or max) price with seasonal adjustment"""
        s = self.awi.current_step
        seller = self.awi.is_first_level
        issues = (
            self.awi.current_output_issues if seller else self.awi.current_input_issues
        )
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value

        # Base price determination
        if best_price:
            base_price = pmax if seller else pmin
        else:
            base_price = random.randint(pmin, pmax)

        # Calculate the seasonal adjustment factor
        # Assuming s is the day of the year (from 0 to 364)
        seasonal_factor = self.seasonal_amplitude * math.sin(
            self.seasonal_frequency * s + self.seasonal_phase_shift
        )
        
        # Adjust the base price with the seasonal factor
        adjusted_price = base_price * (1 + seasonal_factor)
        adjusted_price = max(pmin, min(adjusted_price, pmax))  # Clamp the price within min and max bounds

        return s, int(adjusted_price)

    def thresholds_for_seasons(self):
        seasonal_factor = self.seasonal_amplitude * math.sin(
            self.seasonal_frequency * self.awi.current_step + self.seasonal_phase_shift
        )

        # Adjust 'th' based on seasonal pricing
        self.th = max(0, self.base_th * (1 + seasonal_factor))

        # Adjust 'th_round' to reflect urgency in different seasons
        if seasonal_factor > 0:  # High-demand season
            self.th_round = max(8, self.base_th_round - 4)
        else:  # Low-demand season
            self.th_round = min(20, self.base_th_round + 4)

        # Adjust 'th_over' for excess quantity tolerance
        if seasonal_factor > 0:  # High prices, less tolerance
            self.th_over = max(1, self.base_th_over - 1)
        else:  # Low prices, more tolerance
            self.th_over = self.base_th_over + 1

    def before_step(self):
        self.round = 0
        # Check if the all_partners_info dictionary has been initialized
        if self.awi.current_step == 0:
            if self.awi.is_first_level:
                all_partners = self.awi.my_consumers
            else:
                all_partners = self.awi.my_suppliers

            # Initialize all_partners_info only once, without resetting at every step
            self.all_partners_info = {partner: [0, 0, 0, 0] for partner in all_partners}
        else:
            # Increment the age of data for each partner in subsequent steps
            for p in self.all_partners_info:
                self.all_partners_info[p][3] += 1  # Increment age of data

        return super().before_step()
    

    def on_negotiation_success(self, contract: Contract, mechanism: SAONMI) -> None:
        if self.awi.is_first_level == True:
            partner = contract.annotation["buyer"]
        else:
            partner = contract.annotation["seller"]
        self.all_partners_info[partner][0] += 1  # Number of successful contracts
        self.all_partners_info[partner][1] += contract.agreement[
            "quantity"
        ]  # Contract quantity

    def on_negotiation_failure(
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: SAONMI,
        state: SAOState,
    ) -> None:
        if self.awi.is_first_level == True:
            partner = annotation["buyer"]
        else:
            partner = annotation["seller"]
        self.all_partners_info[partner][2] += (
            1  # Number of unsuccessful contracts(EndNegotiation)
        )


if __name__ == "__main__":
    import sys

    from helpers.runner import run

    run([NewDistRedistAgent], sys.argv[1] if len(sys.argv) > 1 else "oneshot")