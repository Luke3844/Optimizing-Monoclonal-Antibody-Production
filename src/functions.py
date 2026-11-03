# =======================================================================
# FUNCTIONS
# Define reusable helper functions in this file.
# =======================================================================

# Downtime between batches is one day
downtime = 1

def calculate_average(values):
    return sum(values) / len(values)

# Economics Functions
def culture_cost(GlucoseInitial, GlutamineInitial, InitialVCD):
    return 10*GlucoseInitial + 20*GlutamineInitial + 0.0005 * InitialVCD

def culture_value(mAb_final):
    return 1000*mAb_final

def profit(cost, value):
    return value - cost

def space_time_profit(sum_profit, harvest_time):
    harvest_time_d = harvest_time/24.0
    return sum_profit / (harvest_time + downtime)

def space_time_yield(mAb_final, harvest_time):
    harvest_time_d = harvest_time/24.0
    return mAb_final / (harvest_time + downtime)
