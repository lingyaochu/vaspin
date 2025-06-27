"""Module to parse VASP OUTCAR files."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

from vaspin.utils.datatype import SymTensor


class ParserState:
    """Holds state potentially needed by multiple handlers."""

    def __init__(self) -> None:
        """Initialize the parser state."""
        self.num_atoms: int = 0
        self.is_ionic_step_finished: bool = False


class InfoHandler(ABC):
    """Abstract base class for OUTCAR information handlers."""

    KEY: str

    def __init_subclass__(cls):
        """Register subclass in the HANDLERS dictionary."""
        super().__init_subclass__()
        if "KEY" not in cls.__dict__ or not isinstance(cls.KEY, str):
            raise TypeError(
                f"{cls.__name__} must define a class variable 'KEY' of type str."
            )
        HANDLERS[cls.KEY] = cls

    @property
    @abstractmethod
    def HEADER(self) -> str:
        """Header string to identify relevant lines in the OUTCAR file."""
        pass

    def matches(self, line: str) -> bool:
        """Check if the handler should process this line (based on header)."""
        return self.HEADER in line.strip()

    @abstractmethod
    def parse(
        self, lines: List[str], line_idx: int, data: Dict[str, Any], state: ParserState
    ) -> int:
        """Parse the relevant information starting from the matched line.

        Args:
            lines: The full OUTCAR file content as list of lines
            line_idx: Starting line index in the lines list
            data: Dictionary to store parsed data
            state: Parser state object

        Returns:
            The line index after parsing (for continuing the parse)
        """
        pass

    def _log(self, data: Dict[str, Any], message: str):
        """Helper to add logs to a common log list in the data dict."""
        if "_parse_log" not in data:
            data["_parse_log"] = []
        data["_parse_log"].append(f"{self.__class__.__name__}: {message}")

    def _skip_lines(self, lines: List[str], line_idx: int, num_lines: int) -> int:
        """Skip a number of lines and return the new line index."""
        return min(line_idx + num_lines, len(lines))


HANDLERS: Dict[str, Type[InfoHandler]] = {}


class NElectronHandler(InfoHandler):
    """Handles the number of electron in the OUTCAR file."""

    KEY = "N electrons"

    @property
    def HEADER(self) -> str:
        """Set header string for number of electrons."""
        return "total number of electrons"

    def parse(
        self, lines: List[str], line_idx: int, data: Dict[str, Any], state: ParserState
    ) -> int:
        """Parse the number of electrons."""
        self._log(data, "Parsing number of electrons.")
        line = lines[line_idx]
        electrons = float(line.strip().split()[2])
        data["N electrons"] = electrons
        return line_idx + 1


class NIonsHandler(InfoHandler):
    """Handles the number of ions in the OUTCAR file."""

    KEY = "N ions"

    @property
    def HEADER(self) -> str:
        """Set header string for number of ions."""
        return "number of ions     NIONS"

    def parse(
        self, lines: List[str], line_idx: int, data: Dict[str, Any], state: ParserState
    ) -> int:
        """Parse the number of ions."""
        self._log(data, "Parsing number of ions.")
        line = lines[line_idx]
        ions = int(line.strip().split()[-1])
        data["N ions"] = ions
        state.num_atoms = ions
        return line_idx + 1


class DTensorHandler(InfoHandler):
    """Handles the ZFS D tensor block."""

    KEY = "D tensor"

    @property
    def HEADER(self) -> str:
        """Set header string for D tensor block."""
        return "Spin-spin contribution to zero-field splitting tensor (MHz)"

    def parse(
        self, lines: List[str], line_idx: int, data: Dict[str, Any], state: ParserState
    ) -> int:
        """Parse the D tensor block."""
        self._log(data, "Parsing D tensor block.")
        current_idx = self._skip_lines(lines, line_idx, 4)

        if current_idx < len(lines):
            dataline = lines[current_idx]
            parts_sym = [float(x) for x in dataline.strip().split()]
            d_tensor = SymTensor().from_sequence(parts_sym)
            data["D tensor"] = d_tensor.get_matrix_sym().tolist()

        return current_idx + 1


class ForceHandler(InfoHandler):
    """Handles the forces on atoms"""

    KEY = "Forces"

    @property
    def HEADER(self) -> str:
        """Set header string for forces."""
        return "TOTAL-FORCE (eV/Angst)"

    def parse(
        self, lines: List[str], line_idx: int, data: Dict[str, Any], state: ParserState
    ) -> int:
        """Parse the forces on atoms."""
        self._log(data, "Parsing forces on atoms.")
        forces = []
        current_idx = self._skip_lines(lines, line_idx, 1)  # Skip header line

        for _ in range(state.num_atoms):
            if current_idx < len(lines):
                current_idx += 1
                line = lines[current_idx]
                line_parts = line.strip().split()
                force = [float(i) for i in line_parts[3:6]]
                forces.append(force)

        if "forces" not in data:
            data["forces"] = []
        data["forces"].append(forces)
        return current_idx + 1


class IonicEnergyHandler(InfoHandler):
    """Handles the ionic energy in the OUTCAR file."""

    KEY = "Energy"

    @property
    def HEADER(self) -> str:
        """Set header string for ionic energy."""
        return "FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)"

    def parse(
        self, lines: List[str], line_idx: int, data: Dict[str, Any], state: ParserState
    ) -> int:
        """Parse the energy of each ionic step."""
        self._log(data, "Parsing ionic energy.")
        current_idx = self._skip_lines(lines, line_idx, 3)

        if current_idx < len(lines):
            current_idx += 1
            line = lines[current_idx]
            energy = float(line.strip().split()[-1])
            if "Ionic energy" not in data:
                data["Ionic energy"] = []
            data["Ionic energy"].append(energy)

        return current_idx + 1


class PhononHandler(InfoHandler):
    """Handles the phonon frequencies and eigenvectors."""

    KEY = "Phonon"

    @property
    def HEADER(self) -> str:
        """Set header string for phonon frequencies."""
        return "Eigenvectors and eigenvalues of the dynamical matrix"

    def parse(
        self, lines: List[str], line_idx: int, data: Dict[str, Any], state: ParserState
    ) -> int:
        """Parse the phonon frequencies and eigenvectors."""
        self._log(data, "Parsing phonon frequencies.")

        data["phonon"] = {"frequencies": [], "eigenmodes": []}

        current_idx = self._skip_lines(lines, line_idx, 3)

        for _ in range(3 * state.num_atoms):
            # the frequency in THz
            if current_idx < len(lines):
                current_idx += 1
                line = lines[current_idx]
                line_parts = line.strip().split()
                freq = (
                    float(line_parts[3])
                    if line_parts[1] == "f"
                    else -float(line_parts[2])
                )
                data["phonon"]["frequencies"].append(freq)

            # skip one line
            current_idx += 1

            # the eigenvectors
            mode = []
            for _ in range(state.num_atoms):
                if current_idx < len(lines):
                    current_idx += 1
                    line = lines[current_idx]
                    line_parts = line.strip().split()
                    mode.append([float(i) for i in line_parts[3:6]])

            # skip one line
            current_idx += 1
            data["phonon"]["eigenmodes"].append(mode)

        return current_idx


class DielectricEleHandler(InfoHandler):
    """Handles the electron contribution to dielectric tensor."""

    KEY = "Dielectric ele"

    @property
    def HEADER(self) -> str:
        """Set header string for dielectric tensor."""
        return (
            "MACROSCOPIC STATIC DIELECTRIC TENSOR "
            "(including local field effects in DFT)"
        )

    def parse(
        self, lines: List[str], line_idx: int, data: Dict[str, Any], state: ParserState
    ) -> int:
        """Parse the dielectric tensor."""
        # the electronic part appears twice, skip the second
        if "dielectric_ele" in data:
            self._log(data, "The electronic part has been parsed, skip.")
            return line_idx + 1

        self._log(data, "Parsing dielectric tensor, the electronic part.")
        current_idx = self._skip_lines(lines, line_idx, 1)

        for _i in range(3):
            if current_idx < len(lines):
                current_idx += 1
                line = lines[current_idx]
                row = [float(x) for x in line.strip().split()]
                if "dielectric_ele" not in data:
                    data["dielectric_ele"] = []
                data["dielectric_ele"].append(row)

        return current_idx + 1


class DielectricIonHandler(InfoHandler):
    """Handles the ionic contribution to dielectric tensor."""

    KEY = "Dielectric ion"

    @property
    def HEADER(self) -> str:
        """Set header string for dielectric tensor."""
        return "MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC CONTRIBUTION"

    def parse(
        self, lines: List[str], line_idx: int, data: Dict[str, Any], state: ParserState
    ) -> int:
        """Parse the ionic part of dielectric tensor."""
        self._log(data, "Parsing dielectric tensor, the ionic part.")
        current_idx = self._skip_lines(lines, line_idx, 1)

        for _i in range(3):
            if current_idx < len(lines):
                current_idx += 1
                line = lines[current_idx]
                row = [float(x) for x in line.strip().split()]
                if "dielectric_ion" not in data:
                    data["dielectric_ion"] = []
                data["dielectric_ion"].append(row)

        return current_idx + 1


class HyperfineFermiHandler(InfoHandler):
    """Handles the hyperfine interaction parameters from fermi contact."""

    KEY = "Hyperfine fermi"

    @property
    def HEADER(self) -> str:
        """Set header string for hyperfine parameters."""
        return "Fermi contact (isotropic) hyperfine coupling parameter (MHz)"

    def parse(
        self, lines: List[str], line_idx: int, data: Dict[str, Any], state: ParserState
    ) -> int:
        """Parse the hyperfine interaction parameters from fermi contact."""
        self._log(data, "Parsing hyperfine interaction parameters from fermi contact.")
        current_idx = self._skip_lines(lines, line_idx, 3)

        A_pw = []
        A_ps = []
        A_ae = []
        A_c = []
        for _ in range(state.num_atoms):
            if current_idx < len(lines):
                current_idx += 1
                line = lines[current_idx]
                line_parts = line.strip().split()
                A_pw.append(float(line_parts[1]))
                A_ps.append(float(line_parts[2]))
                A_ae.append(float(line_parts[3]))
                A_c.append(float(line_parts[4]))

        data["hyperfine_fermi"] = {"A_pw": A_pw, "A_ps": A_ps, "A_ae": A_ae, "A_c": A_c}
        return current_idx + 1


class HyperfineDipolarHandler(InfoHandler):
    """Handles the hyperfine interaction parameters from dipolar interaction."""

    KEY = "Hyperfine dipolar"

    @property
    def HEADER(self) -> str:
        """Set header string for hyperfine parameters."""
        return "Dipolar hyperfine coupling parameters (MHz)"

    def parse(
        self, lines: List[str], line_idx: int, data: Dict[str, Any], state: ParserState
    ) -> int:
        """Parse the hyperfine interaction parameters from dipolar interaction."""
        self._log(
            data, "Parsing hyperfine interaction parameters from dipolar interaction."
        )
        current_idx = self._skip_lines(lines, line_idx, 3)

        hyper_dipolar = []
        for _ in range(state.num_atoms):
            if current_idx < len(lines):
                current_idx += 1
                line = lines[current_idx]
                line_parts = line.strip().split()
                parts_sym = [float(x) for x in line_parts[1:7]]
                dipolar_tensor = SymTensor().from_sequence(parts_sym)
                hyper_dipolar.append(dipolar_tensor.get_matrix_sym().tolist())

        data["hyperfine_dipolar"] = hyper_dipolar
        return current_idx + 1


class SitePotHandler(InfoHandler):
    """Handles the site potentials for each atoms in the last ionic step"""

    KEY = "Site potential"

    @property
    def HEADER(self) -> str:
        """Set header string for site potentials."""
        return "the norm of the test charge is"

    def parse(
        self, lines: List[str], line_idx: int, data: Dict[str, Any], state: ParserState
    ) -> int:
        """Parse the site potentials for each atom in the last ionic step."""
        self._log(data, "Parsing site potentials for each atom.")

        site_pot = []
        current_idx = self._skip_lines(lines, line_idx, 1)  # Skip the header line

        while current_idx < len(lines):
            line = lines[current_idx]
            line_stripped = line.strip()
            if line_stripped == "":
                break

            parts = line_stripped.split()
            values = [float(parts[i]) for i in range(1, len(parts), 2)]
            site_pot.extend(values)
            current_idx += 1

        data["site_potential"] = site_pot
        return current_idx


class VaspOutcarParser:
    """Parses VASP OUTCAR files using a modular handler system."""

    def __init__(self, outcar_path: str):
        """Initialize the parser with the path to the OUTCAR file."""
        if not isinstance(outcar_path, str):
            raise TypeError("The OUTCAR path must be a string.")

        self.outcar_path: str = outcar_path
        self.data: Dict[str, Any] = {}
        self._state: ParserState = ParserState()
        # Register handler instances
        self._handlers: List[InfoHandler] = self._initialize_handlers()

    def _initialize_handlers(self) -> List[InfoHandler]:
        """Creates and returns a list of handler instances."""
        # The order might matter if some headers are substrings of others,
        # or if some handlers rely on state set by previous ones (like NIONS).
        # Place more specific or prerequisite handlers first.
        return [
            NIonsHandler(),
            NElectronHandler(),
            IonicEnergyHandler(),
            ForceHandler(),
            PhononHandler(),
            DTensorHandler(),
        ]

    def set_handlers(self, handler_keys: List[str]):
        """Set custom handlers to be used for parsing."""
        if len(handler_keys) == 0:
            raise ValueError("At least one handler key must be provided.")
        handlers = []

        for handler_key in handler_keys:
            if handler_key not in HANDLERS:
                raise ValueError(
                    f"Handler '{handler_key}' is not registered."
                    f"Please choose from: {list(HANDLERS.keys())}"
                )
            handlers.append(HANDLERS[handler_key]())

        self._handlers = handlers

    def _find_handler_matches(self, lines: List[str]) -> List[tuple[InfoHandler, int]]:
        """Find all matches for registered handlers in the content."""
        matches = []

        for line_idx, line in enumerate(lines):
            if not line.strip():
                continue

            for handler in self._handlers:
                if handler.matches(line):
                    matches.append((handler, line_idx))
                    break  # Only match the first handler for each line

        return matches

    def parse(self, verbose: bool = False):
        """Parses the OUTCAR file using the registered handlers."""
        self.data = {"_parse_log": []}  # Reset data, initialize log
        self._state = ParserState()  # Reset state

        self._log(f"Starting parsing of '{self.outcar_path}'")

        # Read the entire file content into memory as lines
        with open(self.outcar_path, "r") as f:
            lines = f.readlines()

        # Strip newlines from each line
        lines = [line.rstrip("\n\r") for line in lines]

        self._log(f"File content loaded, {len(lines)} lines")

        # Find all handler matches
        matches = self._find_handler_matches(lines)
        self._log(f"Found {len(matches)} handler matches")

        # Process each match
        for handler, line_idx in matches:
            try:
                handler.parse(lines, line_idx, self.data, self._state)
            except Exception as e:
                self._log(f"Error in {handler.__class__.__name__}: {str(e)}")
                if verbose:
                    import traceback

                    traceback.print_exc()

        self._log("Finished parsing file.")
        if verbose:
            for msg in self.log:
                print(msg)

    def _log(self, message: str):
        """Adds a message to the internal parse log."""
        self.data["_parse_log"].append(f"Parser: {message}")

    @property
    def dmat(self) -> List[List[float]]:
        """Returns the D matrix from the parsed data."""
        return self.data.get("D tensor", [])

    @property
    def force(self) -> List[List[List[float]]]:
        """Returns the forces from the parsed data."""
        return self.data.get("forces", [])

    @property
    def energy(self) -> List[float]:
        """Returns the ionic energy from the parsed data."""
        return self.data.get("Ionic energy", [])

    @property
    def electrons(self) -> float:
        """Returns the number of electrons from the parsed data."""
        return self.data.get("N electrons", 0.0)

    @property
    def ions(self) -> int:
        """Returns the number of ions from the parsed data."""
        return self.data.get("N ions", 0)

    @property
    def phonon(self) -> Dict[str, Any]:
        """Returns the phonon data from the parsed data."""
        return self.data.get("phonon", {})

    @property
    def dielectric_ele(self) -> List[List[float]]:
        """Returns the electronic part of dielectric tensor from the parsed data."""
        return self.data.get("dielectric_ele", [])

    @property
    def dielectric_ion(self) -> List[List[float]]:
        """Returns the ionic part of dielectric tensor from the parsed data."""
        return self.data.get("dielectric_ion", [])

    @property
    def hyperfine_fermi(self) -> Dict[str, List[float]]:
        """Returns the fermi contact hyperfine parameters from the parsed data."""
        return self.data.get("hyperfine_fermi", {})

    @property
    def hyperfine_dipolar(self) -> List[List[List[float]]]:
        """Returns the dipolar hyperfine parameters from the parsed data."""
        return self.data.get("hyperfine_dipolar", [])

    @property
    def site_potential(self) -> List[float]:
        """Returns the site potentials for each atom from the parsed data."""
        return self.data.get("site_potential", [])

    @property
    def log(self) -> List[str]:
        """Returns the parse log."""
        return self.data.get("_parse_log", [])
