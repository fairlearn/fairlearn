import { IFairnessContext } from "./IFairnessContext";

export interface IWizardTabProps {
    dashboardContext: IFairnessContext;
    onNext: () => void;
    onPrevious?: () => void;
}