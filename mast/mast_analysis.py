import subprocess
from enum import Enum
import os.path
from mast_results import parse_results
import mast_writer
import uuid
import examples

TEMP = "./temp/"
MAST_PATH = "./mast-1-5-1-0-bin/"
MAST_EXECUTABLE = "mast_analysis.exe"
MAST_NON_SCHEDULABLE = "Final analysis status: NOT-SCHEDULABLE"
MAST_SCHEDULABLE = "The system is schedulable"


class MastAnalysis(Enum):
    HOLISTIC = "holistic"
    OFFSET = "offset_based_approx"
    OFFSET_OPT = "offset_based_approx_w_pr"


class MastAssignment(Enum):
    NONE = None
    PD = "pd"
    HOSPA = "hospa"


def analyze(system, analysis: MastAnalysis, assignment: MastAssignment):
    # create random temporary file names for this analysis, will be removed afterwards
    name = str(uuid.uuid1())
    input = os.path.abspath(os.path.join(TEMP, name + ".txt"))
    output = os.path.abspath(os.path.join(TEMP, name + "-out.xml"))

    schedulable, results = False, {}
    try:
        # make sure priorities are correct for mast (integers higher than 0)
        mast_writer.sanitize_priorities(system)

        # export system to a file with mast format
        mast_writer.export(system, input)

        # analyze with mast, capture results
        schedulable, results = run(analysis, assignment, input, output)

        # save wcrts into the system
        for task in system.tasks:
            task.wcrt = results[task.name]

        # sanity check: system schedulability must match
        assert system.is_schedulable() == schedulable

    finally:
        # clean-up process: restore original unsanitized priorities, remove temporary files
        mast_writer.desanitize_priorities(system)
        # if os.path.isfile(input):
        #     os.remove(input)
        # if os.path.isfile(output):
        #     os.remove(output)

    return schedulable, results


def run(analysis, assignment, input, output=None):
    cmd = [MAST_EXECUTABLE, analysis.value]
    if assignment is not MastAssignment.NONE:
        cmd.append("-p")
        cmd.append("-t")
        cmd.append(assignment.value)
    cmd.append(input)
    if output:
        cmd.append(output)

    run = subprocess.run(cmd, capture_output=True, cwd=MAST_PATH, shell=True)
    out = run.stdout.decode() if run.stdout else ""

    schedulable = MAST_SCHEDULABLE in out if out else False
    results = parse_results(output) if output and os.path.isfile(output) else {}
    return schedulable, results


if __name__ == '__main__':
    system = examples.get_medium_system()
    # sched = run(MastAnalysis.HOLISTIC, MastAssignment.NONE, "test.txt", "test-out2.xml")
    # print(sched)
    print(analyze(system, MastAnalysis.HOLISTIC, MastAssignment.HOSPA))