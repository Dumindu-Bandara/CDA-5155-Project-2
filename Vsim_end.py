"""
UFID: 61994080
Name: Dumindu Ashen Bandara Elamure Mudiyanselage
Course: CDA 5155 - Computer Architecture
Project 2 - RISC-V Simulator
Date: 2024-11-12

Academic Honesty Statement:
On my honor, I have neither given nor received any unauthorized aid on this assignment.
"""

from enum import Enum
import argparse
import textwrap

MEMORY_START = 256

PRE_ISSUE_BUFFER_SIZE = 4
PRE_ALU1_BUFFER_SIZE = 2
PRE_ALU2_BUFFER_SIZE = 1
PRE_ALU3_BUFFER_SIZE = 1
PRE_MEMORY_BUFFER_SIZE = 1
POST_ALU2_BUFFER_SIZE = 1
POST_ALU3_BUFFER_SIZE = 1
POST_MEMORY_BUFFER_SIZE = 1

MAX_FETCHES_PER_CYCLE = 2
MAX_ISSUES_PER_CYCLE = 3
MAX_ALU1_ISSUES_PER_CYCLE = 1
MAX_ALU2_ISSUES_PER_CYCLE = 1
MAX_ALU3_ISSUES_PER_CYCLE = 1


class DissasemblyState(Enum):
    """State variable which denotes whether the disassembly is in instruction or data mode."""

    INSTRUCTION = 1
    DATA = 2


class InstructionCategory(Enum):
    """Enumeration for instruction categories."""

    CATEGORY_1 = "00"
    CATEGORY_2 = "01"
    CATEGORY_3 = "10"
    CATEGORY_4 = "11"


class Category1Opcode(Enum):
    """Enumeration for Category 1 opcodes."""

    BEQ = "00000"
    BNE = "00001"
    BLT = "00010"
    SW = "00011"


class Category2Opcode(Enum):
    """Enumeration for Category 2 opcodes."""

    ADD = "00000"
    SUB = "00001"
    AND = "00010"
    OR = "00011"


class Category3Opcode(Enum):
    """Enumeration for Category 3 opcodes."""

    ADDI = "00000"
    ANDI = "00001"
    ORI = "00010"
    SLL = "00011"
    SRAI = "00100"
    LW = "00101"


class Category4Opcode(Enum):
    """Enumeration for Category 4 opcodes."""

    JAL = "00000"
    BREAK = "11111"


def twos_complement(bin_str: str) -> int:
    """Convert a binary string in two's complement format to its integer value.

    Args:
        bin_str (str): The binary string to convert.
    Returns:
        int: The integer value of the binary string.
    """

    value = -int(bin_str[0]) * 2 ** (len(bin_str) - 1) + int(bin_str[1:], 2)
    return value


def instruction_decoder(instruction: str, address: int) -> dict[str, int | str | Enum]:
    """Decode a RISC-V instruction into its components.
    Args:
        instruction (str): The binary string representation of the instruction.
        address (int): The memory address of the instruction.
    Returns:
        dict: A dictionary containing the decoded instruction components.
    """

    output_dict = {}

    if instruction[30:32] == InstructionCategory.CATEGORY_1.value:
        opcode = instruction[25:30]
        immediate = instruction[0:7] + instruction[20:25]

        output_dict["immediate"] = twos_complement(immediate)
        output_dict["rs1"] = int(instruction[12:17], 2)
        output_dict["rs2"] = int(instruction[7:12], 2)
        output_dict["func3"] = "000"

        output_dict["category"] = InstructionCategory.CATEGORY_1
        operation = Category1Opcode(opcode)
        output_dict["operation"] = operation

        if operation == Category1Opcode.SW:
            output_dict["assembly"] = (
                f"{instruction}\t{address}\t{operation.name.lower()} x{output_dict['rs1']}, {output_dict['immediate']}(x{output_dict['rs2']})"
            )
        else:
            output_dict["assembly"] = (
                f"{instruction}\t{address}\t{operation.name.lower()} x{output_dict['rs1']}, x{output_dict['rs2']}, #{output_dict['immediate']}"
            )

    elif instruction[30:32] == InstructionCategory.CATEGORY_2.value:
        opcode = instruction[25:30]
        output_dict["rd"] = int(instruction[20:25], 2)
        output_dict["rs1"] = int(instruction[12:17], 2)
        output_dict["rs2"] = int(instruction[7:12], 2)
        output_dict["func3"] = "000"
        output_dict["func7"] = "0000000"

        output_dict["category"] = InstructionCategory.CATEGORY_2
        operation = Category2Opcode(opcode)
        output_dict["operation"] = operation

        output_dict["assembly"] = (
            f"{instruction}\t{address}\t{operation.name.lower()} x{output_dict['rd']}, x{output_dict['rs1']}, x{output_dict['rs2']}"
        )

    elif instruction[30:32] == InstructionCategory.CATEGORY_3.value:
        opcode = instruction[25:30]
        output_dict["rd"] = int(instruction[20:25], 2)
        output_dict["rs1"] = int(instruction[12:17], 2)

        if opcode in [Category3Opcode.SLL.value, Category3Opcode.SRAI.value]:
            immediate = instruction[7:12]
            output_dict["immediate"] = int(immediate, 2)
        else:
            immediate = instruction[0:12]
            output_dict["immediate"] = twos_complement(immediate)

        output_dict["func3"] = "000"

        output_dict["category"] = InstructionCategory.CATEGORY_3
        operation = Category3Opcode(opcode)
        output_dict["operation"] = operation

        if operation == Category3Opcode.LW:
            output_dict["assembly"] = (
                f"{instruction}\t{address}\t{operation.name.lower()} x{output_dict['rd']}, {output_dict['immediate']}(x{output_dict['rs1']})"
            )
        else:
            output_dict["assembly"] = (
                f"{instruction}\t{address}\t{operation.name.lower()} x{output_dict['rd']}, x{output_dict['rs1']}, #{output_dict['immediate']}"
            )

    elif instruction[30:32] == InstructionCategory.CATEGORY_4.value:
        opcode = instruction[25:30]

        output_dict["category"] = InstructionCategory.CATEGORY_4
        operation = Category4Opcode(opcode)
        output_dict["operation"] = operation

        if opcode == Category4Opcode.BREAK.value:
            output_dict["assembly"] = f"{instruction}\t{address}\tbreak"
        elif opcode == Category4Opcode.JAL.value:
            rd = int(instruction[20:25], 2)
            output_dict["rd"] = rd

            immediate = instruction[0:20]
            output_dict["immediate"] = twos_complement(immediate)
            output_dict["assembly"] = (
                f"{instruction}\t{address}\tjal x{output_dict['rd']}, #{output_dict['immediate']}"
            )

    return output_dict


class Disassembler:
    """Class to handle disassembly of RISC-V instructions and data."""

    def __init__(self):
        self.memory = {}
        self.state = DissasemblyState.INSTRUCTION

    def disassemble(self, riscv_text: str):
        """Disassemble the RISC-V instructions and data from the input file.

        Args:
            riscv_text (str): Path to the input file containing RISC-V instructions.
        """

        disassebly_output = []

        with open(riscv_text, "r") as file:
            instructions = file.readlines()

            address = MEMORY_START
            for instruction in instructions:
                instruction = instruction.strip()

                if self.state == DissasemblyState.INSTRUCTION:
                    decoded_instruction = instruction_decoder(instruction, address)
                    disassebly_output.append(decoded_instruction["assembly"])

                    if decoded_instruction["operation"] == Category4Opcode.BREAK:
                        self.state = DissasemblyState.DATA

                elif self.state == DissasemblyState.DATA:
                    data_value = twos_complement(instruction)
                    self.memory[address] = data_value
                    disassebly_output.append(f"{instruction}\t{address} {data_value}")

                address += 4


class ProcessorPipeline:
    def __init__(self, memory):
        """Initialize the processor pipeline with registers, memory, and buffers.

        Args:
            memory (dict): The memory dictionary containing data.
        """

        self.registers = [0] * 32
        self.memory = memory
        self.pc = MEMORY_START
        self.cycle = 1
        self.fetch_waiting = ""
        self.fetch_executed = ""

        # Buffers
        self.pre_issue_prev = []
        self.pre_issue_next = []

        self.alu1_prev = []
        self.alu1_next = []

        self.memory_prev = []
        self.memory_next = []

        self.alu2_prev = []
        self.alu2_next = []

        self.post_alu2_prev = []
        self.post_alu2_next = []

        self.alu3_prev = []
        self.alu3_next = []

        self.post_alu3_prev = []
        self.post_alu3_next = []

        self.post_memory_prev = []
        self.post_memory_next = []

        # Stall Status
        self.fetch_stall_prev = False
        self.fetch_stall_curr = False

        # Pipeline status
        self.ended = False

    def is_branch_raw_exist(self, operand_1, operand_2):
        """Check for RAW hazards for branch instructions. sets fetch_stall_curr if hazard exists.

        Args:
            operand_1 (int): The first source register.
            operand_2 (int): The second source register.
        Returns:
            bool: True if a RAW hazard exists, False otherwise.
        """

        buffers = [
            self.pre_issue_next,
            self.pre_issue_prev,
            self.alu1_prev,
            self.alu2_prev,
            self.alu3_prev,
            self.memory_prev,
            self.post_alu2_prev,
            self.post_alu3_prev,
            self.post_memory_prev,
        ]

        for buffer in buffers:
            for instruction in buffer:
                instruction_dest = instruction.get("rd", None)

                if operand_1 == instruction_dest or operand_2 == instruction_dest:
                    self.fetch_stall_curr = True
                    return True
        else:
            return False

    def instruction_fetch(self):
        """Fetch instructions from memory and handle branch instructions with hazard detection."""

        if self.fetch_stall_prev:
            if self.ended:
                return

            instruction = self.instructions.get(self.pc)
            decoded_instruction = instruction_decoder(instruction, self.pc)

            operand_1 = decoded_instruction["rs1"]
            operand_2 = decoded_instruction["rs2"]

            if self.is_branch_raw_exist(operand_1, operand_2):
                self.fetch_waiting = (
                    "[" + decoded_instruction["assembly"].split("\t")[-1] + "]"
                )
                return
            else:
                # Add branch executed code
                self.fetch_executed = (
                    "[" + decoded_instruction["assembly"].split("\t")[-1] + "]"
                )

                if decoded_instruction["operation"] == Category1Opcode.BEQ:
                    if (
                        self.registers[decoded_instruction["rs1"]]
                        == self.registers[decoded_instruction["rs2"]]
                    ):
                        self.pc += decoded_instruction["immediate"] << 1
                    else:
                        self.pc += 4
                elif decoded_instruction["operation"] == Category1Opcode.BNE:
                    if (
                        self.registers[decoded_instruction["rs1"]]
                        != self.registers[decoded_instruction["rs2"]]
                    ):
                        self.pc += decoded_instruction["immediate"] << 1
                    else:
                        self.pc += 4
                elif decoded_instruction["operation"] == Category1Opcode.BLT:
                    if (
                        self.registers[decoded_instruction["rs1"]]
                        < self.registers[decoded_instruction["rs2"]]
                    ):
                        self.pc += decoded_instruction["immediate"] << 1
                    else:
                        self.pc += 4
                elif decoded_instruction["operation"] == Category4Opcode.BREAK:
                    self.fetch_executed = (
                        "[" + decoded_instruction["assembly"].split("\t")[-1] + "]"
                    )
                    self.ended = True
                    return

        else:
            if self.ended:
                return

            num_issues = min(
                PRE_ISSUE_BUFFER_SIZE - len(self.pre_issue_prev), MAX_FETCHES_PER_CYCLE
            )

            for i in range(num_issues):
                instruction = self.instructions.get(self.pc)
                decoded_instruction = instruction_decoder(instruction, self.pc)

                if decoded_instruction["operation"] == Category4Opcode.BREAK:
                    self.fetch_executed = (
                        "[" + decoded_instruction["assembly"].split("\t")[-1] + "]"
                    )
                    self.ended = True
                    return

                elif decoded_instruction["operation"] in [
                    Category1Opcode.BEQ,
                    Category1Opcode.BNE,
                    Category1Opcode.BLT,
                ]:
                    operand_1 = decoded_instruction["rs1"]
                    operand_2 = decoded_instruction["rs2"]
                    if self.is_branch_raw_exist(operand_1, operand_2):
                        self.fetch_waiting = (
                            "[" + decoded_instruction["assembly"].split("\t")[-1] + "]"
                        )
                        return
                    else:
                        self.fetch_executed = (
                            "[" + decoded_instruction["assembly"].split("\t")[-1] + "]"
                        )

                        if decoded_instruction["operation"] == Category1Opcode.BEQ:
                            if (
                                self.registers[decoded_instruction["rs1"]]
                                == self.registers[decoded_instruction["rs2"]]
                            ):
                                self.pc += decoded_instruction["immediate"] << 1
                            else:
                                self.pc += 4
                            return
                        elif decoded_instruction["operation"] == Category1Opcode.BNE:
                            if (
                                self.registers[decoded_instruction["rs1"]]
                                != self.registers[decoded_instruction["rs2"]]
                            ):
                                self.pc += decoded_instruction["immediate"] << 1
                            else:
                                self.pc += 4
                            return
                        elif decoded_instruction["operation"] == Category1Opcode.BLT:
                            if (
                                self.registers[decoded_instruction["rs1"]]
                                < self.registers[decoded_instruction["rs2"]]
                            ):
                                self.pc += decoded_instruction["immediate"] << 1
                            else:
                                self.pc += 4
                            return

                elif decoded_instruction["operation"] == Category4Opcode.JAL:
                    self.registers[decoded_instruction["rd"]] = self.pc + 4
                    self.pc += decoded_instruction["immediate"] << 1
                    self.fetch_executed = (
                        "[" + decoded_instruction["assembly"].split("\t")[-1] + "]"
                    )
                    return

                else:
                    self.pre_issue_next.append(decoded_instruction)
                    self.pc += 4

    def hazard_detection(self, instruction, issue_buffer_index):
        """Detect hazards for instruction issue. Returns True if a hazard exists, False otherwise."""
        # Hazard detection and issue logic
        # No structural hazards - no speculation - HZ-1
        # No RAW and WAW hazards with active instructions - HZ-2
        # No WAW or WAR hazards with two instructions in the same cycle - HZ-3
        # No WAR hazards with not-issued intructions - HZ-4
        # For MEM all source register are ready - HZ-5
        # Load stay until all previous stores? - HZ-6
        # In order store issue - HZ-7

        earlier_not_issued = self.pre_issue_prev[:issue_buffer_index]

        active_instruction_buffers = [
            self.alu1_prev,
            self.alu1_next,
            self.alu2_prev,
            self.alu2_next,
            self.alu3_prev,
            self.alu3_next,
            self.memory_prev,
            self.post_alu2_prev,
            self.post_alu3_prev,
            self.post_memory_prev,
        ]
        active_instruction_buffers = active_instruction_buffers + [earlier_not_issued]

        same_cycle_issue_buffers = [self.alu1_next, self.alu2_next, self.alu3_next]

        if instruction["operation"] in [Category3Opcode.LW, Category1Opcode.SW]:
            if instruction["operation"] == Category3Opcode.LW:
                operand_1 = instruction.get("rs1", None)
                destination = instruction.get("rd", None)

                # Check RAW for active instructions
                for buffer in active_instruction_buffers:
                    for active_instruction in buffer:
                        active_dest = active_instruction.get("rd", None)

                        if operand_1 == active_dest:
                            return True

                # Check WAW for active instructions # NOTE: This is extreme but let's see
                for buffer in active_instruction_buffers:
                    for active_instruction in buffer:
                        active_dest = active_instruction.get("rd", None)
                        if active_dest == destination:
                            return True

                # Check WAW for same cycle issues
                for buffer in same_cycle_issue_buffers:
                    for same_cycle_instruction in buffer:
                        same_cycle_dest = same_cycle_instruction.get("rd", None)
                        if same_cycle_dest == destination:
                            return True

                # Check WAR for same cycle issues
                for buffer in same_cycle_issue_buffers:
                    for same_cycle_instruction in buffer:
                        same_cycle_operand_1 = same_cycle_instruction.get("rs1", None)
                        same_cycle_operand_2 = same_cycle_instruction.get("rs2", None)

                        if (
                            destination == same_cycle_operand_1
                            or destination == same_cycle_operand_2
                        ):
                            return True

                # Check WAR for not issued instructions
                for not_issued_instruction in earlier_not_issued:
                    not_issued_operand_1 = not_issued_instruction.get("rs1", None)
                    not_issued_operand_2 = not_issued_instruction.get("rs2", None)

                    if (
                        destination == not_issued_operand_1
                        or destination == not_issued_operand_2
                    ):
                        return True

                # Load stay until all previous stores
                for not_issued_instruction in earlier_not_issued:
                    if not_issued_instruction["operation"] == Category1Opcode.SW:
                        return True

            elif instruction["operation"] == Category1Opcode.SW:
                # Check RAW for active instructions
                operand_1 = instruction.get("rs1", None)
                operand_2 = instruction.get("rs2", None)

                for buffer in active_instruction_buffers:
                    for active_instruction in buffer:
                        active_dest = active_instruction.get("rd", None)

                        if operand_1 == active_dest or operand_2 == active_dest:
                            return True

                # In order store issue
                for not_issued_instruction in earlier_not_issued:
                    if not_issued_instruction["operation"] == Category1Opcode.SW:
                        return True

        if instruction["operation"] in [
            Category2Opcode.ADD,
            Category2Opcode.SUB,
            Category3Opcode.ADDI,
        ]:
            # Check RAW for active instructions
            operand_1 = instruction.get("rs1", None)
            operand_2 = instruction.get("rs2", None)
            destination = instruction.get("rd", None)

            for buffer in active_instruction_buffers:
                for active_instruction in buffer:
                    active_dest = active_instruction.get("rd", None)

                    if (operand_1 is not None and operand_1 == active_dest) or (
                        operand_2 is not None and operand_2 == active_dest
                    ):
                        return True

            # Check WAW for active instructions
            for buffer in active_instruction_buffers:
                for active_instruction in buffer:
                    active_dest = active_instruction.get("rd", None)
                    if active_dest is not None and active_dest == destination:
                        return True

            # Check WAW for same cycle issues
            for buffer in same_cycle_issue_buffers:
                for same_cycle_instruction in buffer:
                    same_cycle_dest = same_cycle_instruction.get("rd", None)
                    if same_cycle_dest is not None and same_cycle_dest == destination:
                        return True

            # Check WAR for same cycle issues
            for buffer in same_cycle_issue_buffers:
                for same_cycle_instruction in buffer:
                    same_cycle_operand_1 = same_cycle_instruction.get("rs1", None)
                    same_cycle_operand_2 = same_cycle_instruction.get("rs2", None)

                    if (
                        destination is not None and destination == same_cycle_operand_1
                    ) or (
                        destination is not None and destination == same_cycle_operand_2
                    ):
                        return True

            # Check WAR for not issued instructions
            for not_issued_instruction in earlier_not_issued:
                not_issued_operand_1 = not_issued_instruction.get("rs1", None)
                not_issued_operand_2 = not_issued_instruction.get("rs2", None)

                if (
                    destination is not None and destination == not_issued_operand_1
                ) or (destination is not None and destination == not_issued_operand_2):
                    return True

        if instruction["operation"] in [
            Category2Opcode.AND,
            Category2Opcode.OR,
            Category3Opcode.ANDI,
            Category3Opcode.ORI,
            Category3Opcode.SLL,
            Category3Opcode.SRAI,
        ]:
            operand_1 = instruction.get("rs1", None)
            operand_2 = instruction.get("rs2", None)
            destination = instruction.get("rd", None)

            # Check RAW for active instructions
            for buffer in active_instruction_buffers:
                for active_instruction in buffer:
                    active_dest = active_instruction.get("rd", None)

                    if (operand_1 is not None and operand_1 == active_dest) or (
                        operand_2 is not None and operand_2 == active_dest
                    ):
                        return True

            # Check WAW for active instructions
            for buffer in active_instruction_buffers:
                for active_instruction in buffer:
                    active_dest = active_instruction.get("rd", None)
                    if active_dest is not None and active_dest == destination:
                        return True

            # Check WAW for same cycle issues
            for buffer in same_cycle_issue_buffers:
                for same_cycle_instruction in buffer:
                    same_cycle_dest = same_cycle_instruction.get("rd", None)
                    if same_cycle_dest is not None and same_cycle_dest == destination:
                        return True

            # Check WAR for same cycle issues
            for buffer in same_cycle_issue_buffers:
                for same_cycle_instruction in buffer:
                    same_cycle_operand_1 = same_cycle_instruction.get("rs1", None)
                    same_cycle_operand_2 = same_cycle_instruction.get("rs2", None)

                    if (
                        destination is not None and destination == same_cycle_operand_1
                    ) or (
                        destination is not None and destination == same_cycle_operand_2
                    ):
                        return True

            # Check WAR for not issued instructions
            for not_issued_instruction in earlier_not_issued:
                not_issued_operand_1 = not_issued_instruction.get("rs1", None)
                not_issued_operand_2 = not_issued_instruction.get("rs2", None)

                if (
                    destination is not None and destination == not_issued_operand_1
                ) or (destination is not None and destination == not_issued_operand_2):
                    return True

    def instruction_issue(self):
        """Issue instructions from the pre-issue buffer to the appropriate ALU buffers with hazard detection."""

        issue_count = 0
        alu1_issue_count = 0
        alu2_issue_count = 0
        alu3_issue_count = 0

        pop_index = 0

        while True:
            if issue_count >= MAX_ISSUES_PER_CYCLE or len(self.pre_issue_prev) == 0:
                break

            if pop_index < len(self.pre_issue_prev):
                instruction = self.pre_issue_prev[pop_index]

                is_hazard = self.hazard_detection(instruction, pop_index)
                if is_hazard:
                    pop_index += 1
                    continue

                if instruction["operation"] in [Category3Opcode.LW, Category1Opcode.SW]:
                    if (
                        alu1_issue_count < MAX_ALU1_ISSUES_PER_CYCLE
                        and len(self.alu1_prev) < PRE_ALU1_BUFFER_SIZE
                    ):
                        self.alu1_next.append(self.pre_issue_prev.pop(pop_index))
                        alu1_issue_count += 1
                        issue_count += 1
                    else:
                        pop_index += 1

                if instruction["operation"] in [
                    Category2Opcode.ADD,
                    Category2Opcode.SUB,
                    Category3Opcode.ADDI,
                ]:
                    if (
                        alu2_issue_count < MAX_ALU2_ISSUES_PER_CYCLE
                        and len(self.alu2_prev) < PRE_ALU2_BUFFER_SIZE
                    ):
                        self.alu2_next.append(self.pre_issue_prev.pop(pop_index))
                        alu2_issue_count += 1
                        issue_count += 1
                    else:
                        pop_index += 1

                if instruction["operation"] in [
                    Category2Opcode.AND,
                    Category2Opcode.OR,
                    Category3Opcode.ANDI,
                    Category3Opcode.ORI,
                    Category3Opcode.SLL,
                    Category3Opcode.SRAI,
                ]:
                    if (
                        alu3_issue_count < MAX_ALU3_ISSUES_PER_CYCLE
                        and len(self.alu3_prev) < PRE_ALU3_BUFFER_SIZE
                    ):
                        self.alu3_next.append(self.pre_issue_prev.pop(pop_index))
                        alu3_issue_count += 1
                        issue_count += 1
                    else:
                        pop_index += 1

            else:
                break

        # print("--------- Issue Cycle Debug ---------")
        # print(f"Issued Instructions: {issue_count}")
        # print(f"ALU1 Issues: {alu1_issue_count}")
        # print(f"ALU2 Issues: {alu2_issue_count}")
        # print(f"ALU3 Issues: {alu3_issue_count}")
        # print(f"Fetch Waiting: {self.fetch_waiting}")
        # print(f"Fetch Executed: {self.fetch_executed}")
        # print(f"ALU1 Next: {self.alu1_next}")
        # print(f"ALU1_Prev: {self.alu1_prev}")
        # print(f"ALU2 Next: {self.alu2_next}")
        # print(f"ALU2_Prev: {self.alu2_prev}")
        # print(f"ALU3 Next: {self.alu3_next}")
        # print(f"ALU3_Prev: {self.alu3_prev}")
        # print("-------------------------------------")

    def alu1_execute(self):
        """Supports Category1Opcode.SW and Category3Opcode.LW instructions."""

        if len(self.alu1_prev) == 0:
            return

        instruction = self.alu1_prev.pop(0)

        if instruction["operation"] == Category3Opcode.LW:
            memory_address = (
                self.registers[instruction["rs1"]] + instruction["immediate"]
            )
            instruction["memory_address"] = memory_address

        elif instruction["operation"] == Category1Opcode.SW:
            memory_address = (
                self.registers[instruction["rs2"]] + instruction["immediate"]
            )
            instruction["memory_address"] = memory_address

        self.memory_next.append(instruction)

    def alu2_execute(self):
        """Supports Category2Opcode.ADD, Category2Opcode.SUB, and Category3Opcode.ADDI instructions."""

        if len(self.alu2_prev) == 0:
            return

        instruction = self.alu2_prev.pop(0)

        if instruction["operation"] == Category2Opcode.ADD:
            result = (
                self.registers[instruction["rs1"]] + self.registers[instruction["rs2"]]
            )
            instruction["result"] = result

        elif instruction["operation"] == Category2Opcode.SUB:
            result = (
                self.registers[instruction["rs1"]] - self.registers[instruction["rs2"]]
            )
            instruction["result"] = result

        elif instruction["operation"] == Category3Opcode.ADDI:
            result = self.registers[instruction["rs1"]] + instruction["immediate"]
            instruction["result"] = result

        self.post_alu2_next.append(instruction)

    def alu3_execute(self):
        """Supports Category2Opcode.AND, Category2Opcode.OR, Category3Opcode.ANDI, Category3Opcode.ORI,
        Category3Opcode.SLLI, and Category3Opcode.SRAI instructions."""

        if len(self.alu3_prev) == 0:
            return

        instruction = self.alu3_prev.pop(0)

        if instruction["operation"] == Category2Opcode.AND:
            result = (
                self.registers[instruction["rs1"]] & self.registers[instruction["rs2"]]
            )
            instruction["result"] = result

        elif instruction["operation"] == Category2Opcode.OR:
            result = (
                self.registers[instruction["rs1"]] | self.registers[instruction["rs2"]]
            )
            instruction["result"] = result

        elif instruction["operation"] == Category3Opcode.ANDI:
            result = self.registers[instruction["rs1"]] & instruction["immediate"]
            instruction["result"] = result

        elif instruction["operation"] == Category3Opcode.ORI:
            result = self.registers[instruction["rs1"]] | instruction["immediate"]
            instruction["result"] = result

        elif instruction["operation"] == Category3Opcode.SLL:
            result = self.registers[instruction["rs1"]] << instruction["immediate"]
            instruction["result"] = result

        elif instruction["operation"] == Category3Opcode.SRAI:
            result = self.registers[instruction["rs1"]] >> instruction["immediate"]
            instruction["result"] = result

        self.post_alu3_next.append(instruction)

    def memory_access(self):
        if len(self.memory_prev) == 0:
            return

        instruction = self.memory_prev.pop(0)
        if instruction["operation"] == Category3Opcode.LW:
            instruction["loaded_value"] = self.memory.get(
                instruction["memory_address"], 0
            )
            self.post_memory_next.append(instruction)
        elif instruction["operation"] == Category1Opcode.SW:
            self.memory[instruction["memory_address"]] = self.registers[
                instruction["rs1"]
            ]

    def write_back(self):
        """Write back results to registers from post-memory, post-ALU2, and post-ALU3 buffers."""

        if len(self.post_memory_prev) > 0:
            post_mem_instruction = self.post_memory_prev.pop(0)
            if post_mem_instruction["operation"] == Category3Opcode.LW:
                self.registers[post_mem_instruction["rd"]] = post_mem_instruction[
                    "loaded_value"
                ]

        elif len(self.post_alu2_prev) > 0:
            post_alu2_instruction = self.post_alu2_prev.pop(0)
            self.registers[post_alu2_instruction["rd"]] = post_alu2_instruction[
                "result"
            ]

        elif len(self.post_alu3_prev) > 0:
            post_alu3_instruction = self.post_alu3_prev.pop(0)
            self.registers[post_alu3_instruction["rd"]] = post_alu3_instruction[
                "result"
            ]

    def handle_overflow(self) -> None:
        """Handle overflow for register values to ensure they stay within 32-bit signed integer range."""

        for i in range(len(self.registers)):
            if self.registers[i] < -(2**31):
                self.registers[i] = ((self.registers[i] + 2**31) % 2**32) - 2**31
            elif self.registers[i] > 2**31 - 1:
                self.registers[i] = ((self.registers[i] - 2**31) % 2**32) - 2**31
        return self.registers

    def tick(self):
        """Advance the pipeline by one cycle."""

        # Update previous cycle buffers with next cycle buffers
        self.pre_issue_prev.extend(self.pre_issue_next)
        self.pre_issue_next = []

        self.alu1_prev.extend(self.alu1_next)
        self.alu1_next = []

        self.memory_prev = self.memory_next
        self.memory_next = []

        self.alu2_prev = self.alu2_next
        self.alu2_next = []

        self.post_alu2_prev = self.post_alu2_next
        self.post_alu2_next = []

        self.alu3_prev = self.alu3_next
        self.alu3_next = []

        self.post_alu3_prev = self.post_alu3_next
        self.post_alu3_next = []

        self.post_memory_prev = self.post_memory_next
        self.post_memory_next = []

        self.fetch_stall_prev = self.fetch_stall_curr
        self.fetch_stall_curr = False

        self.registers[0] = 0  # Ensure register x0 is always 0
        self.handle_overflow()  # Handle overflow for registers

    def output_state(self):
        memory_print = ""
        mem_addresses = sorted(self.memory.keys())

        if len(mem_addresses) > 0:
            mem_addresses_min = mem_addresses[0]
            mem_addresses_max = mem_addresses[-1]

            for i in range(mem_addresses_min, mem_addresses_max + 1, 32):
                row = [
                    self.memory.get(addr, 0)
                    for addr in range(i, i + 32, 4)
                    if addr <= mem_addresses_max
                ]

                memory_print += f"{i}:\t" + "\t".join([str(val) for val in row]) + "\n"
        else:
            memory_print = ""

        pre_issue = [
            "[" + instruction["assembly"].split("\t")[-1] + "]"
            for instruction in self.pre_issue_prev
        ]
        pre_alu1 = [
            "[" + instruction["assembly"].split("\t")[-1] + "]"
            for instruction in self.alu1_prev
        ]
        pre_mem = [
            "[" + instruction["assembly"].split("\t")[-1] + "]"
            for instruction in self.memory_prev
        ]
        post_mem = [
            "[" + instruction["assembly"].split("\t")[-1] + "]"
            for instruction in self.post_memory_prev
        ]
        pre_alu2 = [
            "[" + instruction["assembly"].split("\t")[-1] + "]"
            for instruction in self.alu2_prev
        ]
        post_alu2 = [
            "[" + instruction["assembly"].split("\t")[-1] + "]"
            for instruction in self.post_alu2_prev
        ]
        pre_alu3 = [
            "[" + instruction["assembly"].split("\t")[-1] + "]"
            for instruction in self.alu3_prev
        ]
        post_alu3 = [
            "[" + instruction["assembly"].split("\t")[-1] + "]"
            for instruction in self.post_alu3_prev
        ]

        output = (
            textwrap.dedent("""
            {}
            Cycle {}:\n
            IF Unit:
            \tWaiting: {}
            \tExecuted: {}
            Pre-Issue Queue:
            \tEntry 0: {}
            \tEntry 1: {}
            \tEntry 2: {}
            \tEntry 3: {}
            Pre-ALU1 Queue:
            \tEntry 0: {}
            \tEntry 1: {}
            Pre-MEM Queue: {}
            Post-MEM Queue: {}
            Pre-ALU2 Queue: {}
            Post-ALU2 Queue: {}
            Pre-ALU3 Queue: {}
            Post-ALU3 Queue: {}
                            
            Registers
            x00:\t{}
            x08:\t{}
            x16:\t{}
            x24:\t{}
            Data
        """).format(
                "-" * 20,
                self.cycle,
                self.fetch_waiting,  # IF Waiting
                self.fetch_executed,  # IF Executed
                pre_issue[0] if len(pre_issue) > 0 else "",  # Pre-Issue 0
                pre_issue[1] if len(pre_issue) > 1 else "",  # Pre-Issue 1
                pre_issue[2] if len(pre_issue) > 2 else "",  # Pre-Issue 2
                pre_issue[3] if len(pre_issue) > 3 else "",  # Pre-Issue 3
                pre_alu1[0] if len(pre_alu1) > 0 else "",  # Pre-ALU1 0
                pre_alu1[1] if len(pre_alu1) > 1 else "",  # Pre-ALU1 1
                pre_mem[0] if len(pre_mem) > 0 else "",  # Pre-MEM
                post_mem[0] if len(post_mem) > 0 else "",  # Post-MEM
                pre_alu2[0] if len(pre_alu2) > 0 else "",  # Pre-ALU2
                post_alu2[0] if len(post_alu2) > 0 else "",  # Post-ALU2
                pre_alu3[0] if len(pre_alu3) > 0 else "",  # Pre-ALU3
                post_alu3[0] if len(post_alu3) > 0 else "",  # Post-ALU3
                "\t".join(str(self.registers[i]) for i in range(0, 8)),
                "\t".join(str(self.registers[i]) for i in range(8, 16)),
                "\t".join(str(self.registers[i]) for i in range(16, 24)),
                "\t".join(str(self.registers[i]) for i in range(24, 32)),
            )
            + memory_print
        )

        return output

    def process(self, riscv_text: str):
        with open(riscv_text, "r") as file:
            instructions = file.readlines()

            self.instructions = {
                MEMORY_START + i * 4: inst.strip()
                for i, inst in enumerate(instructions)
            }

        with open("simulation_check.txt", "w") as simfile:
            while True:
                self.fetch_waiting = ""
                self.fetch_executed = ""
                self.instruction_fetch()

                self.instruction_issue()
                self.alu1_execute()
                self.alu2_execute()
                self.alu3_execute()
                self.memory_access()
                self.write_back()
                self.tick()

                stop_simulation = False
                active_buffers = [
                    self.pre_issue_prev,
                    self.alu1_prev,
                    self.memory_prev,
                    self.alu2_prev,
                    self.alu3_prev,
                    self.post_alu2_prev,
                    self.post_alu3_prev,
                    self.post_memory_prev,
                ]

                if self.ended:
                    cycle_sim_output = self.output_state()
                    simfile.write(cycle_sim_output[1:])

                    for buffer in active_buffers:
                        if len(buffer) > 0:
                            break
                    else:
                        stop_simulation = True

                if stop_simulation:
                    break

                cycle_sim_output = self.output_state()
                simfile.write(cycle_sim_output[1:])

                self.cycle += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "risv_text", type=str, help="Path to the RISC-V assembly text file"
    )
    args = parser.parse_args()
    riscv_instructions = args.risv_text

    disassembler = Disassembler()
    disassembler.disassemble(riscv_instructions)

    memory = disassembler.memory

    processor = ProcessorPipeline(memory)
    processor.process(riscv_instructions)


if __name__ == "__main__":
    main()
