import collections
import importlib.util
import inspect
import json
import logging
import os
import shlex
from functools import partial
from typing import Dict, List, Mapping, Optional, Union, Any

import yaml
from omegaconf import OmegaConf, DictConfig

from lm_eval.api.group import ConfigurableGroup, GroupConfig
from lm_eval.api.task import ConfigurableTask, Task, TaskConfig

GROUP_ONLY_KEYS = list(GroupConfig().to_dict().keys())

# Setup logging
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
eval_logger = logging.getLogger("lm-eval")


class TaskManager:
    """OmegaConf-native TaskManager with same public interface"""

    def __init__(
        self,
        verbosity="INFO",
        include_path: Optional[Union[str, List]] = None,
        include_defaults: bool = True,
        use_cache: bool = True,
    ) -> None:
        self.verbosity = verbosity
        self.include_path = include_path
        self.use_cache = use_cache
        self.logger = eval_logger
        self.logger.setLevel(getattr(logging, f"{verbosity}"))

        # Setup OmegaConf function constructor
        self._setup_function_constructor()

        # Cache will be set per directory during initialization
        self._cache_data = {}

        # Initialize task and tag registries
        self._task_index, self._tag_registry = self.initialize_tasks(
            include_path=include_path, include_defaults=include_defaults
        )
        self._all_tasks = sorted(list(self._task_index.keys()))

    def _setup_function_constructor(self):
        """Setup YAML constructor for !function tags to keep as strings"""

        def function_constructor(loader, node):
            func_path = loader.construct_scalar(node)
            return f"!function {func_path}"

        yaml.SafeLoader.add_constructor("!function", function_constructor)

    def _get_cache_file_path(self, task_dir):
        return os.path.join(task_dir, ".task_cache")

    def _get_directory_mtime(self, directory):
        """Get the most recent modification time of any file in directory"""
        if not os.path.exists(directory):
            return 0

        latest_mtime = 0
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith((".yaml", ".yml")):
                    file_path = os.path.join(root, file)
                    mtime = os.path.getmtime(file_path)
                    latest_mtime = max(latest_mtime, mtime)
        return latest_mtime

    def _load_cache_for_dir(self, task_dir):
        """Load cache for a directory"""
        cache_file = self._get_cache_file_path(task_dir)
        if not os.path.exists(cache_file):
            return {}

        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)

            # Check if cache is still valid
            dir_mtime = self._get_directory_mtime(task_dir)
            if cache_data.get("directory_mtime", 0) < dir_mtime:
                return {}

            return cache_data.get("data", {})
        except Exception:
            return {}

    def _save_cache_for_dir(self, task_dir, data):
        """Save cache for a directory"""
        if not self.use_cache:
            return

        cache_file = self._get_cache_file_path(task_dir)
        cache_data = {
            "directory_mtime": self._get_directory_mtime(task_dir),
            "data": data,
        }

        try:
            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")

    def initialize_tasks(
        self,
        include_path: Optional[Union[str, List]] = None,
        include_defaults: bool = True,
    ):
        """Initialize task registry and tag registry"""
        task_index = {}
        tag_registry = collections.defaultdict(list)

        # Default task directories
        if include_defaults:
            default_task_dir = os.path.join(os.path.dirname(__file__))
            ti, tr = self._index_tasks_in_directory(default_task_dir)
            task_index.update(ti)
            for tag, tasks in tr.items():
                tag_registry[tag].extend(tasks)

        # Additional directories
        if include_path:
            if isinstance(include_path, str):
                include_path = [include_path]

            for path in include_path:
                if os.path.exists(path):
                    ti, tr = self._index_tasks_in_directory(path)
                    task_index.update(ti)
                    for tag, tasks in tr.items():
                        tag_registry[tag].extend(tasks)

        return task_index, dict(tag_registry)

    def _index_tasks_in_directory(self, directory):
        """Index all tasks and tags in a directory"""
        task_index = {}
        tag_registry = collections.defaultdict(list)

        # Load from cache if available
        if self.use_cache:
            cached_data = self._load_cache_for_dir(directory)
            if cached_data and "tasks" in cached_data and "tags" in cached_data:
                return cached_data["tasks"], cached_data["tags"]

        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith((".yaml", ".yml")):
                    file_path = os.path.join(root, file)
                    try:
                        # For indexing, use simple YAML loading
                        with open(file_path, "r", encoding="utf-8") as f:
                            yaml_content = f.read()

                        # Replace !function to avoid parsing errors during indexing
                        import re

                        yaml_content = re.sub(r"!function\s+\S+", "null", yaml_content)

                        config = yaml.safe_load(yaml_content)

                        # Prioritize group over task when both are present
                        if isinstance(config, dict) and "group" in config:
                            group_name = config["group"]
                            if isinstance(group_name, str):
                                task_index[group_name] = file_path
                        elif isinstance(config, dict) and "task" in config:
                            task_name = config["task"]
                            if isinstance(task_name, str):
                                task_index[task_name] = file_path

                                # Also index any tags
                                if "tag" in config:
                                    tags = config["tag"]
                                    if isinstance(tags, str):
                                        tag_registry[tags].append(task_name)
                                    elif isinstance(tags, list):
                                        for tag in tags:
                                            if isinstance(tag, str):
                                                tag_registry[tag].append(task_name)
                    except Exception as e:
                        self.logger.debug(f"Failed to load {file_path}: {e}")

        # Save to cache
        cache_data = {"tasks": task_index, "tags": dict(tag_registry)}
        self._save_cache_for_dir(directory, cache_data)

        return task_index, dict(tag_registry)

    def _load_yaml_config(self, yaml_path, config_overrides=None):
        """Load YAML config with OmegaConf, keeping functions as strings"""
        try:
            # Load with OmegaConf - functions stay as "!function path" strings
            config = OmegaConf.load(yaml_path)

            # Process includes if present
            if "include" in config:
                config = self._process_includes(config, yaml_path)

            # Store source directory in config for later function resolution
            if "group" not in config:
                config["__source_dir__"] = os.path.dirname(yaml_path)

            # Apply overrides
            if config_overrides:
                if isinstance(config_overrides, str):
                    override_config = OmegaConf.from_dotlist(
                        shlex.split(config_overrides)
                    )
                else:
                    override_config = OmegaConf.create(config_overrides)
                config = OmegaConf.merge(config, override_config)

            return config

        except Exception as e:
            self.logger.error(f"Failed to load config from {yaml_path}: {e}")
            raise

    def _process_includes(self, config, yaml_path):
        """Process include directives by loading and merging included files"""
        yaml_dir = os.path.dirname(yaml_path)

        include_paths = config.include
        if isinstance(include_paths, str):
            include_paths = [include_paths]

        # Remove include from config
        config_without_include = OmegaConf.create(
            {k: v for k, v in config.items() if k != "include"}
        )

        # Load and merge includes in reverse order (so later includes override earlier ones)
        final_config = OmegaConf.create({})
        for include_path in reversed(include_paths):
            if not os.path.isabs(include_path):
                include_path = os.path.join(yaml_dir, include_path)

            if os.path.exists(include_path):
                # Recursively load included config (which may have its own includes)
                include_config = self._load_yaml_config(include_path)
                # Remove source dir from include to avoid conflicts
                include_clean = OmegaConf.create(
                    {k: v for k, v in include_config.items() if not k.startswith("__")}
                )
                final_config = OmegaConf.merge(final_config, include_clean)

        # Merge with main config (main config overrides includes)
        final_config = OmegaConf.merge(final_config, config_without_include)
        return final_config

    def _resolve_functions(self, config_dict, yaml_dir):
        """Resolve function placeholders in config dict"""
        if isinstance(config_dict, dict):
            resolved = {}
            for key, value in config_dict.items():
                resolved[key] = self._resolve_functions(value, yaml_dir)
            return resolved
        elif isinstance(config_dict, list):
            return [self._resolve_functions(item, yaml_dir) for item in config_dict]
        elif isinstance(config_dict, str) and config_dict.startswith("!function "):
            func_path = config_dict[10:]  # Remove "!function "
            return self._load_function(func_path, yaml_dir)
        else:
            return config_dict

    def _load_function(self, func_path, yaml_dir):
        """Load function from module path"""
        try:
            *module_parts, function_name = func_path.split(".")
            module_name = ".".join(module_parts) if module_parts else "utils"
            module_path = os.path.join(yaml_dir, f"{module_name}.py")

            if os.path.exists(module_path):
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return getattr(module, function_name)
            else:
                # Fallback to importing from lm_eval namespace
                module = importlib.import_module(f"lm_eval.{module_name}")
                return getattr(module, function_name)
        except Exception as e:
            self.logger.warning(f"Failed to load function {func_path}: {e}")
            return None

    @property
    def all_tasks(self):
        """Return all available task names"""
        return self._all_tasks

    @property
    def task_index(self):
        """Return task index mapping"""
        return self._task_index

    def match_tasks(self, task_list):
        """Match task patterns against available tasks"""
        import fnmatch

        if isinstance(task_list, str):
            task_list = [task_list]

        matched_tasks = []
        for pattern in task_list:
            matches = fnmatch.filter(self._all_tasks, pattern)
            matched_tasks.extend(matches)

        return list(set(matched_tasks))

    def _name_is_registered(self, name):
        """Check if name is registered"""
        return name in self._task_index

    def _name_is_task(self, name):
        """Check if name is a task"""
        if not self._name_is_registered(name):
            return False
        try:
            config = self._load_yaml_config(self._task_index[name])
            return "task" in config and not "group" in config
        except:
            return False

    def _name_is_group(self, name):
        """Check if name is a group"""
        if not self._name_is_registered(name):
            return False
        try:
            config = self._load_yaml_config(self._task_index[name])
            return "group" in config
        except:
            return False

    def _name_is_tag(self, name):
        """Check if name is a tag"""
        return name in self._tag_registry

    def _name_is_python_task(self, name):
        """Check if name refers to a Python task"""
        if not self._name_is_registered(name):
            return False
        try:
            config = self._load_yaml_config(self._task_index[name])
            return "class" in config
        except:
            return False

    def _config_is_task(self, config):
        """Check if config represents a task"""
        return isinstance(config, dict) and "task" in config and "group" not in config

    def _config_is_group(self, config):
        """Check if config represents a group"""
        return isinstance(config, dict) and "group" in config

    def _config_is_python_task(self, config):
        """Check if config represents a Python task"""
        return isinstance(config, dict) and "class" in config

    def _get_config(self, name):
        """Get config for a registered name"""
        if not self._name_is_registered(name):
            raise ValueError(f"Task/group '{name}' not found")

        return self._load_yaml_config(self._task_index[name])

    def _get_tasklist(self, name):
        """Get task list for a tag"""
        if name in self._tag_registry:
            return self._tag_registry[name]
        return -1

    def load_task_or_group(
        self,
        task_list: Optional[Union[str, list]] = None,
        config_overrides: Optional[Union[str, DictConfig]] = None,
    ) -> dict:
        """Load tasks/groups with OmegaConf-native resolution"""
        if isinstance(task_list, str):
            task_list = [task_list]

        all_loaded_tasks = {}
        for task_spec in task_list:
            loaded_tasks = self._load_individual_task_or_group(
                task_spec, config_overrides
            )
            all_loaded_tasks.update(loaded_tasks)

        return all_loaded_tasks

    def load_config(self, config, config_overrides: Optional[Union[str, DictConfig]] = None):
        resolved_configs = self._resolve_config(config, config_overrides)
        return self._create_task_objects(resolved_configs)

    def _load_individual_task_or_group(self, spec: str,  config_overrides: Optional[Union[str, DictConfig]] = None) -> dict:
        """Load individual task or group preserving hierarchical structure"""
        if self._name_is_task(spec):
            # Single task - return flat structure
            resolved_configs = self._resolve_spec(spec, config_overrides)
            return self._create_task_objects(resolved_configs)

        elif self._name_is_group(spec):
            # Group - preserve hierarchy
            group_config = OmegaConf.to_container(self._get_config(spec))
            group = ConfigurableGroup(group_config)

            # Recursively load subtasks
            group_contents = {}
            tasks = group_config.pop("task", [])
            for subtask_spec in tasks:
                if isinstance(subtask_spec, str):
                    merged_overrides = self._resolve_config(group_config, config_overrides)[0]
                    subtask_result = self._load_individual_task_or_group(subtask_spec, merged_overrides)
                    group_contents.update(subtask_result)
                elif isinstance(subtask_spec, dict):
                    # Inline subgroup: merge parent group defaults (group_config after `task` pop)
                    # with this dict, same idea as _resolve_group_config (lines ~500–506).
                    merged = OmegaConf.merge(
                        OmegaConf.create(group_config),
                        OmegaConf.create(subtask_spec),
                    )
                    merged_container = OmegaConf.to_container(merged)
                    nested_group = ConfigurableGroup(merged_container)
                    nested_tasks = merged_container.pop("task", [])
                    nested_contents = {}
                    for nested_sub in nested_tasks:
                        merged_overrides = self._resolve_config(merged_container, config_overrides)[0]
                        nested_sub_result = self._load_individual_task_or_group(
                            nested_sub, merged_overrides
                        )
                        nested_contents.update(nested_sub_result)
                    group_contents.update({nested_group: nested_contents})
                else:
                    raise TypeError(f"Invalid subtask spec type: {type(subtask_spec)}")
            return {group: group_contents}

        elif self._name_is_tag(spec):
            # Tag - return flat structure of all tagged tasks
            resolved_configs = self._resolve_spec(spec, config_overrides)
            return self._create_task_objects(resolved_configs)

        else:
            raise ValueError(f"Unknown spec: {spec}")

    def _resolve_spec(self, spec: str, global_overrides=None) -> List[Dict]:
        """Resolve a spec (task/group/tag) into list of task configs"""
        if self._name_is_task(spec):
            config = self._get_config(spec)
            if global_overrides:
                # Apply overrides using OmegaConf
                if isinstance(global_overrides, str):
                    override_config = OmegaConf.from_dotlist(
                        shlex.split(global_overrides)
                    )
                else:
                    override_config = OmegaConf.create(global_overrides)
                config = OmegaConf.merge(config, override_config)
            return [config]

        elif self._name_is_group(spec):
            return self._resolve_group(spec, global_overrides)

        elif self._name_is_tag(spec):
            return self._resolve_tag(spec, global_overrides)

        else:
            raise ValueError(f"Unknown spec: {spec}")

    def _resolve_config(self, config_dict, global_overrides=None) -> List:
        """Resolve a config dict"""
        config = OmegaConf.create(config_dict)

        if global_overrides:
            # Apply overrides using OmegaConf
            if isinstance(global_overrides, str):
                override_config = OmegaConf.from_dotlist(shlex.split(global_overrides))
            else:
                override_config = OmegaConf.create(global_overrides)
            config = OmegaConf.merge(config, override_config)

        if self._config_is_group(config):
            return self._resolve_group_config(config, global_overrides)
        else:
            return [config]

    def _resolve_group(self, group_name: str, global_overrides=None) -> List:
        """Resolve group into list of task configs"""
        group_config = self._get_config(group_name)
        return self._resolve_group_config(group_config, global_overrides)

    def _resolve_group_config(self, group_config, global_overrides=None) -> List:
        """Resolve group config into task configs"""
        # Split group config into group metadata and task defaults
        group_meta = {k: v for k, v in group_config.items() if k in GROUP_ONLY_KEYS}
        task_defaults = {
            k: v for k, v in group_config.items() if k not in GROUP_ONLY_KEYS
        }
        task_defaults_omega = OmegaConf.create(task_defaults)

        # Preserve source directory from group config
        group_source_dir = group_config.get("__source_dir__", ".")

        resolved_tasks = []
        for subtask_spec in group_meta.get("task", []):
            if isinstance(subtask_spec, str):
                # Recursively resolve subtask
                subtask_configs = self._resolve_spec(subtask_spec, global_overrides)
            else:
                # Direct config
                subtask_configs = [OmegaConf.create(subtask_spec)]

            # Apply group inheritance using OmegaConf
            for subtask_config in subtask_configs:
                if not isinstance(subtask_config, DictConfig):
                    subtask_config = OmegaConf.create(subtask_config)

                merged = OmegaConf.merge(task_defaults_omega, subtask_config)

                # Ensure source directory is preserved (prefer subtask's source, fallback to group's)
                if "__source_dir__" not in merged and group_source_dir:
                    merged["__source_dir__"] = group_source_dir

                if global_overrides:
                    if isinstance(global_overrides, str):
                        override_config = OmegaConf.from_dotlist(
                            shlex.split(global_overrides)
                        )
                    else:
                        override_config = OmegaConf.create(global_overrides)
                    merged = OmegaConf.merge(merged, override_config)

                resolved_tasks.append(merged)

        return resolved_tasks

    def _resolve_tag(self, tag_name: str, global_overrides=None) -> List:
        """Resolve tag into list of task configs"""
        task_list = self._get_tasklist(tag_name)
        if task_list == -1:
            raise ValueError(f"Unknown tag: {tag_name}")

        resolved_tasks = []
        for task_name in task_list:
            task_configs = self._resolve_spec(task_name, global_overrides)
            resolved_tasks.extend(task_configs)

        return resolved_tasks

    def _create_task_objects(self, configs: List) -> Dict[str, Task]:
        """Create task objects from resolved configs - resolve functions here"""
        tasks = {}

        for config in configs:
            # Get source directory from DictConfig
            if isinstance(config, DictConfig) and "__source_dir__" in config:
                source_dir = config.__source_dir__
            else:
                source_dir = "."

            task_name = config.get("task", "unknown")

            # Filter out group-only keys, internal keys, and include directives
            # Convert to plain Python containers to handle DictConfig/ListConfig
            task_config_dict = OmegaConf.to_container(config, resolve=False)
            task_config_dict = {
                k: v
                for k, v in task_config_dict.items()
                if k not in {"aggregate_metric_list", "group_alias"}
                and not k.startswith("__")
            }

            # NOW resolve functions right before task creation

            resolved_task_config = self._resolve_functions(task_config_dict, source_dir)

            if "class" in resolved_task_config:
                # Python task
                task_class = resolved_task_config["class"]
                if inspect.isclass(task_class) and hasattr(task_class, "__init__"):
                    # Check if constructor accepts config
                    sig = inspect.signature(task_class.__init__)
                    if "config" in sig.parameters:
                        task_obj = task_class(config=resolved_task_config)
                    else:
                        task_obj = task_class()
                else:
                    task_obj = task_class

                # Set task name for ConfigurableTask
                if isinstance(task_obj, ConfigurableTask):
                    task_obj.config.task = task_name
            else:
                # YAML task - functions are resolved here
                task_obj = ConfigurableTask(config=resolved_task_config)

            tasks[task_name] = task_obj

        return tasks


def get_task_name_from_config(task_config):
    """Extract task name from config"""
    if isinstance(task_config, dict):
        return task_config.get("task", "unknown")
    return "unknown"


def get_task_name_from_object(task_object):
    """Extract task name from task object"""
    if hasattr(task_object, "config") and hasattr(task_object.config, "task"):
        return task_object.config.task
    elif hasattr(task_object, "_config") and isinstance(task_object._config, dict):
        return task_object._config.get("task", "unknown")
    return "unknown"


def _check_duplicates(task_dict):

    """Check for duplicate task names"""
    seen = set()
    duplicates = set()

    for task in task_dict:
        name = get_task_name_from_object(task) if hasattr(task, "config") else str(task)
        if name in seen:
            duplicates.add(name)
        seen.add(name)

    if duplicates:
        raise ValueError(f"Duplicate task names found: {duplicates}")


def get_task_dict(
    task_name_list: List[Union[str, Dict, Task]],
    task_manager: TaskManager = None,
    verbosity="INFO",
    include_path=None,
    include_defaults=True,
    use_cache=True,
    config_overrides=None,
):
    """Get task dictionary - main entry point"""
    if task_manager is None:
        task_manager = TaskManager(
            verbosity=verbosity,
            include_path=include_path,
            include_defaults=include_defaults,
            use_cache=use_cache,
        )

    # Process mixed input types
    task_names = []
    direct_configs = []

    for item in task_name_list:
        if isinstance(item, str):
            task_names.append(item)
        elif isinstance(item, dict):
            direct_configs.append(item)
        else:
            # Assume it's already a task object
            task_name = get_task_name_from_object(item)
            return {task_name: item}

    # Load tasks from names
    task_dict = {}
    if task_names:
        task_dict.update(task_manager.load_task_or_group(task_names, config_overrides))

    # Load tasks from direct configs
    for config in direct_configs:
        config_tasks = task_manager.load_config(config, config_overrides)
        task_dict.update(config_tasks)

    _check_duplicates(task_dict)
    return task_dict
