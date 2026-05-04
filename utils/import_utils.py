import importlib
import os
import pkgutil

def find_all_modules(path: str, base_package: str):
    """
    Recursively find all modules in a path.
    """
    modules = []
    for loader, module_name, is_pkg in pkgutil.walk_packages([path], base_package + '.'):
        modules.append(module_name)
    return modules

def find_packages(root_dir: str):
    """
    Find all relevant packages in the root directory.
    """
    packages = []
    for item in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, item)) and os.path.exists(os.path.join(root_dir, item, '__init__.py')):
            if item in ['system', 'model', 'guidance', 'representations', 'inversion', 'sdedit']:
                packages.append(item)
    return packages

def load_all_registries(root_dir: str = None):
    """
    Automatically import all modules in relevant packages to trigger registration.
    """
    if root_dir is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if we've already loaded registries to avoid circular imports and overhead
    if hasattr(load_all_registries, '_loaded'):
        return
    load_all_registries._loaded = True
    
    relevant_packages = find_packages(root_dir)
    for pkg in relevant_packages:
        path = os.path.join(root_dir, pkg)
        for module_name in find_all_modules(path, pkg):
            try:
                importlib.import_module(module_name)
            except Exception as e:
                # You might want to log this instead of just passing
                print(f"Failed to import {module_name}: {e}")
                pass
