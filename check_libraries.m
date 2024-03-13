% List of required toolboxes including image libraries
requiredToolboxes = {'Statistics and Machine Learning Toolbox', 'Deep Learning Toolbox', 'Image Processing Toolbox'};

% Get the installed toolboxes
installedToolboxes = ver;  % This will return an array of structures with information about each installed toolbox

% Check for the presence of required toolboxes and report
for i = 1:length(requiredToolboxes)
    % Look for the required toolbox in the list of installed toolboxes
    isInstalled = any(strcmp({installedToolboxes.Name}, requiredToolboxes{i}));
    
    % Display the result
    if isInstalled
        fprintf('%s is installed.\n', requiredToolboxes{i});
    else
        fprintf('%s is NOT installed.\n', requiredToolboxes{i});
    end
end
